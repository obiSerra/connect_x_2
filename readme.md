```
sudo docker build -t connect-x .
```

```
rm -rf logs/connect_x.log && sudo docker run -it  -v ./connect_x:/app/connect_x -v ./logs:/app/logs connect-x
```