
# note

* this directory is in a premature state
* user interface will change over time
* a quick info for those who want to use it now

# setup to see images

(can be skipped when you do not need to see classified images)

* generate data by doing
```
$ cd ../data
$ make -f data.mk
```

* move data from ../data/ by
```
$ mv ../data/imgs .
```

# run

```
$ cd ..
$ ./vgg.g++
```

# see the log

```
$ ./parse_log.py ../vgg.log
```

open index.html with your browser

you should be able to see

 * how the loss function evolved over time
 * how much time is spent on which kernel
 * history of classifications

