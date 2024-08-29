
# Written: Barbaros Cetiner(ImHandler in old BRAILS)
#          minor edits for BRAILS++ by fmk 03/24
# license: BSD-3 (see LICENSCE.txt file: https://github.com/NHERI-SimCenter/BrailsPlusPlus)

from brails.scrapers.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory

import math
import json
import requests
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import linemerge, unary_union, polygonize
from shapely.strtree import STRtree
from brails.utils.geo_tools import *
import concurrent.futures
from requests.adapters import HTTPAdapter, Retry
import unicodedata
import geopandas as gpd
import geopy.geocoders as geopy


class OSM_FootprintScraper(FootprintScraper):
    """
    A class to generate the foorprint data utilizing Open Street Maps API

    Attributes:

    Methods:
        __init__: Constructor that just creates an empty footprint
        get_inventory(id, coordinates): to get the inventory

    """

    def _cleanstr(self, inpstr):
        return "".join(
            char
            for char in inpstr
            if not char.isalpha()
            and not char.isspace()
            and (char == "." or char.isalnum())
        )

    def _yearstr2int(self, inpstr):
        if inpstr != "NA":
            yearout = self._cleanstr(inpstr)
            yearout = yearout[:4]
            if len(yearout) == 4:
                try:
                    yearout = int(yearout)
                except:
                    yearout = None
            else:
                yearout = None
        else:
            yearout = None

        return yearout

    def _height2float(self, inpstr, lengthUnit):

        if inpstr != "NA":
            heightout = self._cleanstr(inpstr)
            try:
                if lengthUnit == "ft":
                    heightout = round(float(heightout) * 3.28084, 1)
                else:
                    heightout = round(float(heightout), 1)
            except:
                heightout = None
        else:
            heightout = None

        return heightout

    def __init__(self, input: dict):
        """
        Initialize the object

        Args
            input: a dict defining length units, if no ;length' ft is assumed
        """

        self.lengthUnit = input.get("length")
        if self.lengthUnit == None:
            self.lengthUnit = "ft"

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        This method will be used by the caller to obtain the footprints for builings 
        in an area.

        Args:
            region (RegionBoundary): The region of interest.

        Returns:
            BuildingInventory: A building inventory for buildings in the region.

        """

        bpoly, queryarea_printname, osmid = region.get_boundary()

        if osmid != None:

            queryarea_turboid = osmid + 3600000000
            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            area({queryarea_turboid})->.searchArea;
            way["building"](area.searchArea);
            out body;
            >;
            out skel qt;
            """

        else:
            bpoly, queryarea_printname = self.__bbox2poly(queryarea)

            if len(queryarea) == 4:
                bbox = [
                    min(queryarea[1], queryarea[3]),
                    min(queryarea[0], queryarea[2]),
                    max(queryarea[1], queryarea[3]),
                    max(queryarea[0], queryarea[2]),
                ]
                bbox = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            elif len(queryarea) > 4:
                bbox = 'poly:"'
                for i in range(int(len(queryarea) / 2)):
                    bbox += f"{queryarea[2*i+1]} {queryarea[2*i]} "
                bbox = bbox[:-1] + '"'

            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            way["building"]({bbox});
            out body;
            >;
            out skel qt;
            """

        url = "http://overpass-api.de/api/interpreter"
        r = requests.get(url, params={"data": query})

        datalist = r.json()["elements"]
        nodedict = {}
        for data in datalist:
            if data["type"] == "node":
                nodedict[data["id"]] = [data["lon"], data["lat"]]

        attrmap = {
            "start_date": "erabuilt",
            "building:start_date": "erabuilt",
            "construction_date": "erabuilt",
            "roof:shape": "roofshape",
            "height": "buildingheight",
        }

        levelkeys = {"building:levels", "roof:levels", "building:levels:underground"}
        otherattrkeys = set(attrmap.keys())
        datakeys = levelkeys.union(otherattrkeys)

        attrkeys = ["buildingheight", "erabuilt", "numstories", "roofshape"]
        attributes = {key: [] for key in attrkeys}
        fpcount = 0
        footprints = []
        for data in datalist:
            if data["type"] == "way":
                nodes = data["nodes"]
                footprint = []
                for node in nodes:
                    footprint.append(nodedict[node])
                footprints.append(footprint)

                fpcount += 1
                availableTags = set(data["tags"].keys()).intersection(datakeys)
                for tag in availableTags:
                    nstory = 0
                    if tag in otherattrkeys:
                        attributes[attrmap[tag]].append(data["tags"][tag])
                    elif tag in levelkeys:
                        try:
                            nstory += int(data["tags"][tag])
                        except:
                            pass

                    if nstory > 0:
                        attributes["numstories"].append(nstory)
                for attr in attrkeys:
                    if len(attributes[attr]) != fpcount:
                        attributes[attr].append("NA")

        attributes["buildingheight"] = [
            self._height2float(height, self.lengthUnit)
            for height in attributes["buildingheight"]
        ]

        attributes["erabuilt"] = [
            self._yearstr2int(year) for year in attributes["erabuilt"]
        ]

        attributes["numstories"] = [
            nstories if nstories != "NA" else None
            for nstories in attributes["numstories"]
        ]

        print(
            f"\nFound a total of {fpcount} building footprints in {queryarea_printname}"
        )

        return self._create_asset_inventory(footprints, attributes, self.lengthUnit)


    def get_footprints_coordlist(self, lat, lon, address_list, 
                                 basic_geoloc_method='GoogleV3') -> AssetInventory:
        """
        This method returns the footprints of building per lat lon lists
    
        Args:
            lat: (list)
                list of latitude coordinate for properties of interest
            long: (list)
                list of longitude coordinate for properties of interest
            address_list: (list)
                list of addresses for properties of interest
            basic_geoloc_method: (str)
                name of the geocoding method used to generate the lat lon lists
    
        Returns:
            BuildingInventory: A building inventory for buildings in the region.
    
        """                
        
        attrkeys = ["buildingheight", "erabuilt", "numstories", "roofshape"]
        attributes = {key: [] for key in attrkeys}
        fpbldgscount = 0
        footprints = []
        geoloc_options = [basic_geoloc_method, 'ArcGIS', 'Nominatim', 'Photon'] # alternative methods if basic does not work
        geo_flag = True
        invalid_id = [] # list of properties that did not get a valid footprint
        
        for bldg_i in range(len(lat)):            
            #print(address_list[bldg_i] + ', bldg_i = ' + str(bldg_i))
            
            if np.isnan(lat[0]):
                # no geolocation
                footprints.append('NA')
                for attr in attrkeys:
                    attributes[attr].append('NA')
                print('Coordinates are NaN')                             
            
            for geoloc_method in geoloc_options:
                #print(geoloc_method)
        
                # Geolocate again if not basic method
                if geoloc_method != basic_geoloc_method:
                    match geoloc_method:
                        case 'ArcGIS':
                            geolocator = geopy.ArcGIS(timeout = 10)      
                        case 'Nominatim':
                            geolocator = geopy.Nominatim(user_agent="my_app", timeout = 10) 
                        case 'Photon':
                            geolocator = geopy.Photon(timeout = 10)
                    try:
                        location = geolocator.geocode(address_list[bldg_i], exactly_one=False)
                        lat[bldg_i], lon[bldg_i] = location[0].latitude, location[0].longitude
                    except:
                        location = None
                    
                    # check geolocation gives one unique property
                    if (location is None) or (len(location) > 1):
                        geo_flag = False
                    else:
                        geo_flag = True
                
                if geo_flag:
                        
                    # Call OSM API for building 50m around the given point
                    # query = f"""
                    #     [out:json][timeout:5000][maxsize:2000000000];
                    #     way(around:50,{lat[bldg_i]},{lon[bldg_i]})[building];
                    #     out body;
                    #     >;
                    #     out skel qt;
                    #     """
                
                    # Call OSM API at the lat lon
                    query = f"""
                        [out:json][timeout:5000][maxsize:2000000000];
                        (
                          is_in({lat[bldg_i]},{lon[bldg_i]});
                          area._[building];
                        );
                        out body; 
                        >; 
                        out skel qt;
                        """
                
                    # Call OSM API for building 50m around the given point
                    # query = f"""
                    # [out:json][timeout:5000][maxsize:2000000000];
                    # (
                    #   is_in({lat[bldg_i]},{lon[bldg_i]})->.searchArea;
                    #   area.searchArea[building];
                    # );
                    # out body; 
                    # >; 
                    # out skel qt;
                    # """
                    
                    url = "http://overpass-api.de/api/interpreter"
                    r = requests.get(url, params={"data": query})
                    
                    datalist = r.json()["elements"]
                    nodedict = {}
                    for data in datalist:
                        if data["type"] == "node":
                            nodedict[data["id"]] = [data["lon"], data["lat"]]
                    
                    attrmap = {
                        "start_date": "erabuilt",
                        "building:start_date": "erabuilt",
                        "construction_date": "erabuilt",
                        "roof:shape": "roofshape",
                        "height": "buildingheight",
                    }
                    
                    levelkeys = {"building:levels", "roof:levels", "building:levels:underground"}
                    otherattrkeys = set(attrmap.keys())
                    datakeys = levelkeys.union(otherattrkeys)
                                        
                    # keep only footprints with a building tag. If none, then consider all avaialble footprints
                    datalist_updated = []
                    for data in datalist:
                        if ("tags" in data.keys()) and ("building" in data["tags"].keys()):
                            datalist_updated.append(data)
                    if datalist_updated:
                        datalist = datalist_updated
                    
                    # Re organize the fetched footprints
                    fpcount = 0
                    footprint = []
                    footprints_bldg = [] 
                    attributes_bldg = {key: [] for key in attrkeys}
                    
                    for data in datalist:
                        
                        # Identify if the footprint is a building
                        if (data["type"] == "way"):
                            if ("tags" in data.keys()) and (not "area" in data["tags"].keys()) and (not "leisure" in data["tags"].keys()) and (not "landuse" in data["tags"].keys()):
                                is_bldg_fp = True
                            elif ("tags" not in data.keys()):
                                is_bldg_fp = True
                            else:
                                is_bldg_fp = False
                        else:
                            is_bldg_fp = False
                            
                        # if building, collect it
                        if is_bldg_fp:
                            nodes = data["nodes"]
                            footprint = []
                            for node in nodes:
                                footprint.append(nodedict[node])
                            footprints_bldg.append(footprint)
                    
                            fpcount += 1
        
                            if ("tags" in data.keys()):
                                availableTags = set(data["tags"].keys()).intersection(datakeys)
                                for tag in availableTags:
                                    nstory = 0
                                    if tag in otherattrkeys:
                                        attributes_bldg[attrmap[tag]].append(data["tags"][tag])
                                    elif tag in levelkeys:
                                        try:
                                            nstory += int(data["tags"][tag])
                                        except:
                                            pass
                
                                    if nstory > 0:
                                        attributes_bldg["numstories"].append(nstory)
                        
                            # if no tags
                            for attr in attrkeys:
                                if len(attributes_bldg[attr]) != fpcount:
                                    attributes_bldg[attr].append("NA")                                    
                    
                    # Check if footprints were returned
                    if footprint:
                        fpbldgscount += 1
                        break # do not try more geolocation methods. This returned a footprint

            # Review if footprints were returned with all the geolocation trials
            if (footprint) and (geo_flag):
                # Discard all the footprints that do no intercept with our point and keep the one with centroid 
                # closest to our coordinates (the top in the list)
                
                #Turn building footprints into geopandas 
                fp_list = []
                for fp_i in range(len(footprints_bldg)):
                    fp_list.append(Polygon(footprints_bldg[fp_i]))
                gdf = gpd.GeoDataFrame(geometry=fp_list)
            
                if len(gdf) > 1:
                    # Turn building coordinates into geopandas
                    point = [Point(lon[bldg_i], lat[bldg_i])]
                    gdf_point = gpd.GeoDataFrame(geometry=point)
                
                    # Get the index of the intercepting polygons with the lat lon coordinates
                    intersecting_polygon = gpd.sjoin(gdf_point, gdf, predicate="within")
                    fp_idx = intersecting_polygon['index_right'].to_numpy()
                    fp_idx = np.min(fp_idx) # keep the top fp (lowest index) since that is the closest to the desired point
                else:
                    fp_idx = 0
                
                # Store the results for the one footprint
                footprints.append(footprints_bldg[fp_idx])
                for attr in attrkeys:
                    attributes[attr].append(attributes_bldg[attr][fp_idx])
            else:
                # No footprint returned
                footprints.append([[0,0], [0,0]])
                for attr in attrkeys:
                    attributes[attr].append('NA')
                print(address_list[bldg_i] + ', bldg_i = ' + str(bldg_i))
                print('   at '+str(lat[bldg_i]) + ' ,' + str(lon[bldg_i]) + ' do not overlap with a footprint')
                invalid_id.append(bldg_i)
                
        # Save in the proper format for AssetInventory class
        attributes["buildingheight"] = [
            self._height2float(height, self.lengthUnit)
            for height in attributes["buildingheight"]
        ]
        
        attributes["erabuilt"] = [
            self._yearstr2int(year) for year in attributes["erabuilt"]
        ]
        
        attributes["numstories"] = [
            nstories if nstories != "NA" else None
            for nstories in attributes["numstories"]
        ]
        
        print(
            f"\nFound a total of {fpbldgscount} building footprints"
        )
        
        inventory = self._create_asset_inventory(footprints, attributes, self.lengthUnit)
        
        # Dict with properties with and without footprints from this set
        invalid_properties = {}
        invalid_properties['address_list'] = np.array(address_list)[invalid_id]
        invalid_properties['lat'] = np.array(lat)[invalid_id]
        invalid_properties['lon'] = np.array(lon)[invalid_id]
        
        valid_properties = {}
        valid_properties['address_list'] = np.delete(np.array(address_list), invalid_id)
        valid_properties['lat'] = np.delete(np.array(lat), invalid_id)
        valid_properties['lon'] = np.delete(np.array(lon), invalid_id)
        
        return inventory, valid_properties, invalid_properties
