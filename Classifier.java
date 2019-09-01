package com.example.kubra.project3;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;


public class Classifier {

    private static final String TAG = "TfLite";
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE =1;
    private static final int  DIM_HEIGHT =28;
    private static final int DIM_WIDTH = 28;
    private static final int BYTES =4;

    protected Interpreter tflite;
    public char[] class_name = new char[47];
    private static int digit = -1;
    private static float  prob = 0.0f;

    protected ByteBuffer imgData = null;
    private float[][] ProbArray = null;
    protected String ModelFile = "model.tflite";

    Classifier(Activity activity) throws IOException {
        tflite = new Interpreter(loadModelFile(activity));
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_PIXEL_SIZE * BYTES);
        imgData.order(ByteOrder.nativeOrder());
        ProbArray = new float[1][47];
        Log.d(TAG, " Tensorflow Lite Classifier.");
        class_name = new char[]{'J', 'K', 'L', 'M', 'N', 'O', 'Z', 'n', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'i', 'p', 'Q', 'R', 'S', 'T', 'u', 'v', 'w', 'x', 'y', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'q', 'r', 't'};
    }
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(ModelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void classifyMat(Mat mat) {

        long startTime = SystemClock.uptimeMillis();
        if(tflite!=null) {

            convertMattoTfLiteInput(mat);
            runInference();
        }

        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer and run inference " + Long.toString(endTime - startTime));
    }
    private void convertMattoTfLiteInput(Mat mat)
    {
        imgData.rewind();
        float pixel = (float) 0.0;
        for (int i = 0; i < DIM_HEIGHT; ++i) {
            for (int j = 0; j < DIM_WIDTH; ++j) {
              //  imgData.putFloat((float)mat.get(i,j)[0]);

                if(mat.get(i,j)[0]>0.5){
                    pixel = (float) 0.0;
                    Log.d(TAG, "1 yapildi " );

                }
                else{
                    pixel = (float) 1.0;
                    Log.d(TAG, "0 yapildi " );

                }

                imgData.putFloat(pixel);
            }
        }
    }

    private void runInference() {
        Log.e(TAG, "Inference doing");
        if(imgData != null)
            tflite.run(imgData, ProbArray);
        Log.e(TAG, "Inference done "+maxProbIndex(ProbArray[0]));
    }
    private  int maxProbIndex(float[] probs) {
        int maxIndex = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }
        prob = maxProb;
        digit = maxIndex;
        return maxIndex;
    }
    public char getCharacter()
    {

        return class_name[digit];
    }
    public float getProb()
    {

        return prob;
    }
    public void close() {
        if(tflite!=null)
        {
            tflite.close();
            tflite = null;
        }

    }
}