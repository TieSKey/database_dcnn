=========================================
Database Embedding DCNN for CBIR 
==========================================

Repository for CS 645 final project

Imagenet images should be added to the images folder under a 'imagenet' directory. This will be automatically excluded from git by the gitignore file. 

You will need to generate the caffemodel for alexnet and the caffe reference network. You can find the script to generate them in the caffe installation directory. 

GNU General Public License

-------------------------------------------
Python Modules Needed 

sudo pip install cython

sudo pip install hickle

sudo pip install psycopg2

sudo pip install scikit-learn

sudo pip install lshash==0.0.4dev

--------------------------------------------
functions using python

1. sudo apt-get install postgresql-contrib postgresql-plpython postgresql-server-dev-9.4

2. use dnn.sql to implement functions

3. Distance functions: http://pgsimilarity.projects.pgfoundry.org/ (Install, add config to postgresql.conf, create estension in server)
