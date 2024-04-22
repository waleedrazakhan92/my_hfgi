**Note** This repository aims to return the "nearest image" and not the "nearest person" in comparison with the inout face image.

The version that returns the nearest image can be found in: https://github.com/waleedrazakhan92/face_recognition

# Face Recognition Process

For face recognition you need to have a folder with all the images in it.
The directory structure to create a database should look like this:
```
All Images/
│    ├───img1.png
│    ├───img2.png
│    ├───img3.png
│        │
│        │
│        │
```
First you need to run the script *create_face_database.py*. This would generate two numpy(*npy) files. One numpy file stores all the encodings of the images and the other stores the corresponding paths of the images for these encodings.

## Create face database
```
python3 create_face_database.py --path_dataset 'path/to/images/directory/' \
--savename_encodings 'encodings.npy' \
--savename_encodings_paths 'imgs_paths.npy' \
--save_dir 'dataset_encodings/' 
```
##### Arguments (create_face_database)
The function arguments for **create_face_database.py** file are as follows:  

```
--path_dataset              Path to folder containing all the images
--savename_encodings        Name of the numpy file which saves encodings
--savename_encodings_paths  Name of the numpy file which saves encoded images paths
--save_dir                  Directory to save both numpy files
```

## Register New Face
Now if you want to register new faces to the database(i.e two ".npy" files), you need to register_new_face.py 
The directory structure is the same as the first one. You now just need to provide the path of the folder with new images to add to database.
```
python3 register_new_face.py --path_new_person 'path/to/new images/' \
--path_encodings_old 'dataset_encodings/encodings.npy' \
--path_encodings_paths_old 'dataset_encodings/imgs_paths.npy' \
--save_dir 'dataset_encodings/' \
--savename_encodings_updated 'new_encodings.npy' \
--savename_encodings_paths_updated 'new_encoding_paths.npy' 
```

##### Arguments (register_new_face)
The function arguments for **register_new_face.py** file are as follows:  

```
--path_new_person                       Path to folder containing new images
--savename_encodings_updated            Updated name of the numpy file which saves encodings
--savename_encodings_paths_updated      Updated name of the numpy file which saves encoded images paths
--save_dir                              Directory to save both numpy files
--path_encodings_old                    Path of old encodings
--path_encodings_paths_old              Path of old encoding paths
```
This will give updated database which you can use further.

## Register Multiple Faces

For this script the directory structure need to be like this:
```
Dataset Folder/
    ├───Folder 1/
    │   ├───img1.ext
    │   ├───img2.ext
    │   ├───img3.ext
    ├───Folder 2/
    │   ├───img1.ext
    │   ├───img2.ext
    │   ├───img3.ext
    ├───Folder 3/
    │   ├───img1.ext
    │   ├───img2.ext
    │   ├───img3.ext
            │
            │
            │
```

If you want to register multiple folders of images then you need to run the *register_multiple_faces.py* script. 

```
python3 register_multiple_faces.py --path_new_faces 'path/to/new faces directory/' \
--path_encodings_old 'dataset_encodings/encodings.npy'\
--path_encodings_paths_old 'dataset_encodings/imgs_paths.npy' \
--save_dir 'dataset_encodings/' \
--savename_encodings_updated 'new_encodings.npy' \
--savename_encodings_paths_updated 'new_encoding_paths.npy' 
```

##### Arguments (register_multiple_faces)
The function arguments for **register_multiple_faces.py** file are as follows:  

```
--path_new_faces                        Path to folder containing new images
--savename_encodings_updated            Updated name of the numpy file which saves encodings
--savename_encodings_paths_updated      Updated name of the numpy file which saves encoded images paths
--save_dir                              Directory to save both numpy files
--path_encodings_old                    Path of old encodings
--path_encodings_paths_old              Path of old encoding paths
```

## Find Matching Faces
Once the face registration process is complete, you can use the *find_matching_faces.py* file to find the matching faces and store them in a folder.

```
python find_matching_faces.py --img_path 'path/to/image.png' \
--encodings 'dataset_encodings/encodings.npy' \
--encoding_paths 'dataset_encodings/imgs_paths.npy' \
--max_imgs 3 \
--threshold 0.5
```
##### Arguments (find_matching_faces)
The arguments for **find_matching_faces.py** file are as follows:
```
--img_path                  Path of the query image
--path_results              Path to save results            
--encodings                 Path of the encodings.npy file
--encoding_paths            Path of the encoding paths
--threshold                 Threshold distance for a face to be considered as matched
--max_imgs                  Maximum number of images to return within the set threshold
```
Once this code is run, you're going to get the matching images in a separate folder.
