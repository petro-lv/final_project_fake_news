# Fake News Detection Demo


## Build an Image
```
docker build . -t gradio_demo
```

## Create a Container

```
docker run --rm -d -p 8080:8080 --user=42420:42420 gradio_demo
```