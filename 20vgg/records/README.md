
* generate data by doing
```
cd ../data
make -f data.mk
```

* move data from ../data/ by
```
mv ../data/imgs .
```

* copy ../data/cifar-10-batches-bin/batches.meta.txt and add header line "class"
```
cp ../data/cifar-10-batches-bin/batches.meta.txt .
add "class" in the first line
```
