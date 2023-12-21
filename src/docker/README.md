# Dockerfiles


How to build (CUDA):
```bash
sudo docker build . -f docker/era_5g_train_detection_standalone/gpu.Dockerfile -t but5gera/era_5g_train_detection_standalone:VERSION
```

How to run:
```bash
sudo docker run --gpus all --rm -p 5896:5896 but5gera/era_5g_train_detection_standalone:VERSION
```

## Related Repositories

For further information, see [5G-Era Reference-NetApp](https://github.com/5G-ERA/Reference-NetApp) and the corresponding readme on Dockerfiles there (mainly the section about era_5g_object_detection_standalone).


