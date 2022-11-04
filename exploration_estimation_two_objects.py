import numpy as np
import pandas as pd 
from shapely.geometry import Point 
from shapely.geometry.polygon import Polygon 
import cv2 


def inaccurate_to_na(df, p):
    """ Sets all values in the dataframe (with bodypart coordinates) lower than the p threshold to NA

    Parameters
    ----------
    df : pd.Dataframe
        The dataframe with results from DLC
    p : float
        The threshold below which coordinates of bodyparts are set to NA

    Returns
    -------
    dataframe
        The dataframe with results from DLC without inaccurate coordinates
    """

    for bodypart in ["nose", "center"]: 
        df['x_' + bodypart] = [np.nan if l <
                               p else x_coord for (l, x_coord) in zip(df['l_' + bodypart], df['x_'+bodypart])]
        df['y_' + bodypart] = [np.nan if l <
                               p else y_coord for (l, y_coord) in zip(df['l_' + bodypart], df['y_'+bodypart])]
    return df


def create_polygon(x, y, offset): 
    """ Creates a polygon from corner coordinates and scales it with the offset (unless offset is 0)

    Parameters
    ----------
    x : float
        X coordinates of corner coordinates
    y : float
        Y coordinates of corner coordinates
    offset: int
        Amount of pixels the polygon is scaled with

    Returns
    -------
    Polygon
        The polygon that is created in this function
    2D array
        Coordinates from the (scaled) polygon
    """
    coords_object = list(zip(x, y))
    poly_object = Polygon(coords_object)
    poly_object = poly_object.buffer(offset)
    x, y = poly_object.exterior.coords.xy
    return poly_object, np.column_stack((x, y))


def in_object(row, poly_object, poly_object_scaled): 
    """ Evaluates if the nose is close to the object and center is not in the object (thus if animal is exploring)

    Parameters
    ----------
    row : dict
        A row in the dataframe (a frame of the video to be evaluated)
    poly_object : Polygon
        The unscaled object (object1/object2)
    poly_object_scaled : Polygon
        The scaled object (object1/object2)

    Returns
    -------
    Bool
        True is mouse is exploring, False if not
    """
    point_nose = Point(row['x_nose'], row['y_nose']) 
    point_center = Point(row['x_center'], row['y_center'])
    if not poly_object.contains(point_center):
        return poly_object_scaled.contains(point_nose)
    else:
        return False

def cummulative_time(col, fps): 
    """ Calculates commulative time that a value is True in a column

    Parameters
    ----------
    col : series
        A new column of a dataframe

    Returns
    -------
    series
        A column that shows the cummulative time that a value was True
    """
    col = col.cumsum() 
    col = col/fps 
    return col


def write_video(animal, video, outpath, df, fps, coords_object1_scaled, coords_object2_scaled, coords_object1_og, coords_object2_og): 
    """ Creates video that shows the detection of object exploration 

    Parameters
    ----------
    animal : int
        The animal number
    video:  path 
        The path to the video 
    outpath: path
        The path the video should be saved in
    df: pd.Dataframe
        The dataframe created by deeplabcut and modified in get_info
    fps: float
        The frames per second of the video
    coords_object1_scaled, coords_object2_scaled, coords_object1_og, coords_object2_og: 2D array
        The coordinates of all polygons of objects
    """ 

    df['x_nose'] = df['x_nose'].fillna(0) 
    df['y_nose'] = df['y_nose'].fillna(0)
    df['x_center'] = df['x_center'].fillna(0)
    df['y_center'] = df['y_center'].fillna(0)

    video = cv2.VideoCapture(video) 

    df['cum_time_object1'] = cummulative_time(df['in_object1'],fps)
    df['cum_time_object2'] = cummulative_time(df['in_object2'], fps)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    res = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath+animal+'_result'+'.mp4', fourcc, fps, res)

    ret, frame = video.read() 

    while(ret):

        frame_nr = video.get(cv2.CAP_PROP_POS_FRAMES) 

        font = cv2.FONT_HERSHEY_SIMPLEX

        text_object1 = "Object1: " + \
            str(df.loc[df.index[round(frame_nr-1)], 'cum_time_object1']) + "s  "
        text_object2 = "Object2: " + \
            str(df.loc[df.index[round(frame_nr-1)], 'cum_time_object2']) + "s  "

        cv2.putText(frame, text_object1, (25, 30), font,
                    0.75, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, text_object2, (200, 30), font,
                    0.75, (0, 0, 255), 2, cv2.LINE_4)

        cv2.polylines(frame, np.int32(
            [coords_object1_scaled]), isClosed=True, color=(0, 255, 255), thickness=1)
        cv2.polylines(frame, np.int32(
            [coords_object2_scaled]), isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.polylines(frame, np.int32(
            [coords_object1_og]), isClosed=True, color=(0, 0, 0), thickness=1)
        cv2.polylines(frame, np.int32(
            [coords_object2_og]), isClosed=True, color=(0, 0, 0), thickness=1)

        cv2.circle(frame, (round(df.loc[df.index[round(frame_nr-1)], 'x_nose']), round(
            df.loc[df.index[round(frame_nr-1)], 'y_nose'])), 1, (0, 255, 0), 1)
        cv2.circle(frame, (round(df.loc[df.index[round(frame_nr-1)], 'x_center']), round(
            df.loc[df.index[round(frame_nr-1)], 'y_center'])), 1, (255, 255, 255), 1)

        out.write(frame)
        ret, frame = video.read()

    video.release()
    out.release()

    cv2.destroyAllWindows()


def get_info(animal, p, offset, df, fps, coords_object1, coords_object2, video, outpath, make_video): 
    """ Takes in DLC output dataframe and modifies it to show in which frames the animal is exploring, 
    also creates video to display this.

    animal: int
        The animalnumber
    p : float
        The threshold below which coordinates of bodyparts are set to NA
    offset: int
        Amount of pixels the polygon is scaled with
    df : pd.Dataframe
        The dataframe created by deeplabcut and modified in get_info
    fps: float
        The frames per second of the video
    coords_object1, coords_object2: csv file
        Csv files containing the coordinates from the corners of both polygon objects
    video: str
        The path to the video file
    make_video: bool, optional
        A flag used to indicate if a video that shows the exploration detection should be detected (default is True)

    Returns
    -------
    pd.Dataframe
        A dataframe that containing all extra information (how long each object is explored)
    """
    df = df.rename(columns={'coords': 'frame', 'x': 'x_nose', 'y': 'y_nose', 'likelihood': 'l_nose',
                            'x.1': 'x_center', 'y.1': 'y_center', 'likelihood.1': 'l_center'})

    poly_object1_og, coords_object1_og = create_polygon(coords_object1.X, coords_object1.Y, 0)
    poly_object2_og, coords_object2_og = create_polygon(coords_object2.X, coords_object2.Y, 0)

    poly_object1_scaled, coords_object1_scaled = create_polygon(
        coords_object1.X, coords_object1.Y, offset)  
    poly_object2_scaled, coords_object2_scaled = create_polygon(
        coords_object2.X.copy(), coords_object2.Y.copy(), offset)

    df = inaccurate_to_na(df, p)
    """This bottom section calls the in_object function, and says true if in_object returns True"""

    df['in_object1'] = df.apply(in_object, args=(
        poly_object1_og, poly_object1_scaled), axis=1)
    df['in_object2'] = df.apply(in_object, args=(
        poly_object2_og, poly_object2_scaled), axis=1)
    
    if make_video:
        write_video(animal, video, outpath, df, fps, coords_object1_scaled, coords_object2_scaled, coords_object1_og, coords_object2_og)
   
    return df


def aggregated_info(df, fps, results, animal, inter):
    """ Create a dataframe with all summary results from the analysis

    Parameters
    ----------
    df : pd.Dataframe
        The dataframe created by deeplabcut and modified in get_info
    fps: float
        The frames per second of the video
    results: pd.Dataframe
        The dataframe to which summary results form all animals are added 
    animal: int
        The animalnumber
    inter: int
        The total exploration time is also calculated until this frame number

    Returns
    -------
    pd.Dataframe
        A dataframe with summary results from an animal added to it
    """

    total_time_object1 = len(df[df['in_object1']]) / fps
    total_time_object2 = len(df[df['in_object2']]) / fps

    inter_time_object1 = len(df[df['in_object1']].loc[:inter]) / fps
    inter_time_object2 = len(df[df['in_object2']].loc[:inter]) / fps

    l_average = sum(df['l_nose'])/len(df['l_nose'])
    results=results.append({'animal':animal, 'object1':total_time_object1, 'object2':total_time_object2,
                   'inter_object1':inter_time_object1, 'inter_object2':inter_time_object2, 'l_average':l_average}, ignore_index=True)

    return results

