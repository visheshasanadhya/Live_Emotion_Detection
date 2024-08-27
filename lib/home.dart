import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_v2/tflite_v2.dart';

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  late CameraImage? cameraImage;
  late CameraController? cameraController;
  String output = '';
  bool isModelBusy = false; // Track the status of the interpreter

  @override
  void initState() {
    super.initState();
    loadModel();
    loadCamera();
  }

  loadModel() async {
    await Tflite.loadModel(
      model: 'assets/model.tflite',
      labels: 'assets/labels.txt',
      numThreads: 1,
    );
  }

  loadCamera() async {
    final cameras = await availableCameras();
    if (cameras.isNotEmpty) {
      cameraController = CameraController(cameras[0], ResolutionPreset.medium);
      await cameraController!.initialize();
      setState(() {
        cameraController!.startImageStream((CameraImage image) {
          setState(() {
            cameraImage = image;
          });
          runModel();
        });
      });
    } else {
      // Handle the case where no cameras are available
      print('No cameras available');
    }
  }

  runModel() async {
    // Check if the interpreter is busy, exit if it is
    if (isModelBusy) {
      return;
    }
    isModelBusy = true; // Mark interpreter as busy
    if (cameraImage != null && cameraImage!.planes.isNotEmpty) {
      List<dynamic>? predictions = await Tflite.runModelOnFrame(
        bytesList: cameraImage!.planes.map((plane) {
          return plane.bytes;
        }).toList(),
        imageHeight: cameraImage!.height,
        imageWidth: cameraImage!.width,
        imageMean: 0,
        imageStd: 255,
        rotation: 0,
        numResults: 2,
        threshold: 0.1,
        asynch: true,
      );
      if (predictions != null && predictions.isNotEmpty) {
        setState(() {
          output = predictions[0]['label'];
        });
      }
    }
    isModelBusy = false; // Mark interpreter as not busy after inference is completed
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Emotion Test'),
      ),
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.all(20),
            child: Container(
              height: MediaQuery.of(context).size.height * 0.7,
              width: MediaQuery.of(context).size.width,
              child: !cameraController!.value.isInitialized
                  ? Container()
                  : AspectRatio(
                aspectRatio: cameraController!.value.aspectRatio,
                child: CameraPreview(cameraController!),
              ),
            ),
          ),
          Text(output, style: TextStyle(fontSize: 20)),
        ],
      ),
    );
  }

  @override
  void dispose() {
    cameraController?.dispose();
    Tflite.close();
    super.dispose();
  }
}