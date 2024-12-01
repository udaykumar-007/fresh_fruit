package com.example.fruit_app_2;

import ai.onnxruntime.*;
import android.content.Context;
import android.graphics.*;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ObjectDetection {

    private OrtEnvironment env;
    private OrtSession session;
    private float confidenceThreshold;
    private float iouThreshold;



    public ObjectDetection(Context context, String modelPath, float confidenceThreshold, float iouThreshold) throws OrtException {
        this.confidenceThreshold = confidenceThreshold;
        this.iouThreshold = iouThreshold;


        System.out.println(confidenceThreshold);
        // Initialize ONNX Runtime environment and session
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.addCPU(true);
        session = env.createSession(modelPath, options);
    }

    public List<float[]> runInference(Bitmap inputBitmap) throws OrtException {
        // Preprocess the image
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, 640, 640, true);
        FloatBuffer inputBuffer = preprocessImage(resizedBitmap);

        // Prepare the input tensor
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer, new long[]{1, 3, 640, 640});

        // Run inference
        OrtSession.Result result = session.run(Collections.singletonMap(session.getInputNames().iterator().next(), inputTensor));

        // Get the output
        float[][][] output = (float[][][]) result.get(0).getValue();

        // Transpose the output for easier processing
        float[][] transposedOutput = transposeOutput(output[0]);
        // Post-process the output
        return postprocess(inputBitmap, transposedOutput);
    }

    private FloatBuffer preprocessImage(Bitmap bitmap) {
        int modelWidth = 640, modelHeight = 640;
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, modelWidth, modelHeight, false);
        FloatBuffer buffer = FloatBuffer.allocate(3 * modelWidth * modelHeight);

        int[] pixels = new int[modelWidth * modelHeight];
        resizedBitmap.getPixels(pixels, 0, modelWidth, 0, 0, modelWidth, modelHeight);

        for (int channel = 0; channel < 3; channel++) {
            for (int i = 0; i < pixels.length; i++) {
                int pixel = pixels[i];
                float value = 0;

                if (channel == 0) value = ((pixel >> 16) & 0xFF) / 255.0f; // Red channel
                if (channel == 1) value = ((pixel >> 8) & 0xFF) / 255.0f;  // Green channel
                if (channel == 2) value = (pixel & 0xFF) / 255.0f;         // Blue channel

                buffer.put(value);
            }
        }
        buffer.rewind();
        return buffer;
    }

    private float[][] transposeOutput(float[][] batchOutput) {
        int numPredictions = batchOutput.length;
        int numFeatures = batchOutput[0].length;

        float[][] transposedOutput = new float[numFeatures][numPredictions];
        for (int i = 0; i < numPredictions; i++) {
            for (int j = 0; j < numFeatures; j++) {
                transposedOutput[j][i] = batchOutput[i][j];
            }
        }

        return transposedOutput;
    }

    private List<float[]> postprocess(Bitmap inputImage, float[][] transposedOutput) {
        int imgWidth = inputImage.getWidth();
        int imgHeight = inputImage.getHeight();
        if (confidenceThreshold < 0.1f) confidenceThreshold =0.1f;

        List<float[]> detections = new ArrayList<>();
        for (float[] detection : transposedOutput) {
            float confidence = 0;
            int classId = -1;

            // Find the best class and its confidence
            for (int i = 5; i < detection.length; i++) {
                if (detection[i] > confidence) {
                    confidence = detection[i];
                    classId = i - 5;
                }
            }

            if (confidence > confidenceThreshold) {
                float xCenter = detection[0];
                float yCenter = detection[1];
                float width = detection[2];
                float height = detection[3];
                detections.add(new float[]{xCenter, yCenter, width, height, confidence, classId});
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        List<float[]> finalDetections = applyNMS(detections);


        return finalDetections;
    }

    private List<float[]> applyNMS(List<float[]> detections) {
        List<float[]> result = new ArrayList<>();
        Collections.sort(detections, (a, b) -> Float.compare(b[4], a[4])); // Sort by confidence descending

        while (!detections.isEmpty()) {
            float[] best = detections.remove(0);
            result.add(best);

            detections.removeIf(det -> computeIoU(best, det) > iouThreshold);
        }

        return result;
    }

    private float computeIoU(float[] boxA, float[] boxB) {
        float x1 = Math.max(boxA[0] - boxA[2] / 2, boxB[0] - boxB[2] / 2);
        float y1 = Math.max(boxA[1] - boxA[3] / 2, boxB[1] - boxB[3] / 2);
        float x2 = Math.min(boxA[0] + boxA[2] / 2, boxB[0] + boxB[2] / 2);
        float y2 = Math.min(boxA[1] + boxA[3] / 2, boxB[1] + boxB[3] / 2);

        float intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float areaA = boxA[2] * boxA[3];
        float areaB = boxB[2] * boxB[3];
        float union = areaA + areaB - intersection;

        return intersection / union;
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }



}