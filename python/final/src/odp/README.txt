1. Download files from this link and use the makefile to create .vw.gz datasets
https://github.com/JohnLangford/vowpal_wabbit/tree/master/demo/recall_tree/odp
2. Unzip the .gz file and get .vw files
3. Use save_to_tfrecords in util.py to save the dataset into tfrecords format
4. Specify the location of tfrecords in odp_demo.py and run with command
    python3 odp_demo.py [B] [R] [gpu_id]
