package com.example.kubra.project3;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.BackgroundSubtractor;

import org.opencv.video.Video;

import java.io.IOException;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    TextView tv;
    static final String TAG = MainActivity.class.getSimpleName();
    private JavaCameraView cameraView;
    public Mat source,gray_img,resized_img,mask;
    String result;
    private Classifier classifier;
    private Mat mRgba;
    private Mat intermediate;
    private Mat CNN_input;

    private BaseLoaderCallback openCVLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {

            switch (status) {

                case LoaderCallbackInterface.SUCCESS:

                    Log.i(TAG, "OpenCV loaded successfully");
                    cameraView.enableView();
                    cameraView.setVisibility(View.VISIBLE);
                    break;

                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        tv = (TextView)findViewById(R.id.textView);
        cameraView = findViewById(R.id.camera);
        cameraView.setVisibility(View.VISIBLE);
        cameraView.setCvCameraViewListener(this);

    }

    @Override
    protected void onPause() {
        super.onPause();

        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        try {
            classifier = new Classifier(MainActivity.this);
        } catch (IOException e) {
            e.printStackTrace();
        }


        if (!OpenCVLoader.initDebug()) {

            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, openCVLoaderCallback);

        } else {

            Log.d(TAG, "OpenCV library found inside package. Using it!");
            openCVLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {

        if (cameraView != null) {
            cameraView.disableView();
        }

        super.onDestroy();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
     /*   source = new Mat();
        gray_img = new Mat();
        resized_img = new Mat();
        mask = new Mat();
*/
        mRgba = new Mat();
        intermediate = new Mat();
        CNN_input = new Mat();

    }

    @Override
    public void onCameraViewStopped() {
/*        source.release();
        gray_img.release();
        resized_img.release();
        mask.release();*/
        if(intermediate!=null)
            intermediate.release();
        if(CNN_input!=null)
            CNN_input.release();
        if(mRgba!=null)
            mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

       /* source = inputFrame.rgba();
        gray_img = inputFrame.gray();
        Core.rotate(gray_img,gray_img,Core.ROTATE_90_CLOCKWISE);
        Core.rotate(source,source,Core.ROTATE_90_CLOCKWISE);
        Imgproc.resize(gray_img,gray_img, new org.opencv.core.Size(28,28));
        Imgproc.threshold(gray_img,gray_img, 20, 255,Imgproc.THRESH_BINARY_INV);

        //  Imgproc.dilate(gray_img,gray_img,Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));
        return gray_img;*/
        mRgba=inputFrame.rgba();
        Core.rotate(mRgba,mRgba,Core.ROTATE_90_CLOCKWISE);

        int top = mRgba.rows()/2 - 140;
        int left = mRgba.cols() / 2 - 140;
        int height = 140*2;
        int width = 140*2;
        Mat topcorner;

        ///prepocess frame


        Mat gray = inputFrame.gray();
        Core.rotate(gray,gray,Core.ROTATE_90_CLOCKWISE);

        //draw cropped region
        Imgproc.rectangle(mRgba, new Point(mRgba.cols()/2 - 150, mRgba.rows() / 2 - 150), new Point(mRgba.cols() / 2 + 150, mRgba.rows() / 2 + 150), new Scalar(255,255,255),1);
        //crop frame
        Mat graytemp = gray.submat(top, top + height, left, left + width);
        //blur the cropped frame to remove noise
        Imgproc.GaussianBlur(graytemp, graytemp, new org.opencv.core.Size(7,7),2 , 2);

        //convert gray frame to binary using apadative thresold
        Imgproc.threshold(graytemp,intermediate, 127, 255,Imgproc.THRESH_BINARY_INV);

//        Imgproc.adaptiveThreshold(graytemp, intermediate, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 5, 5);
        /*Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(3,3));
        //dilate the frame
        Imgproc.dilate(intermediate, intermediate, element1);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(3,3));
        //erode the frame
        Imgproc.erode(intermediate, intermediate, element);
*/
        Imgproc.resize(intermediate, CNN_input, new org.opencv.core.Size(28,28));///CNN input
        //Log.v("CNN_input size ", CNN_input.rows()+" X "+CNN_input.cols()+" X "+CNN_input.channels()+" "+CNN_input.type());
        //show preprocessed cropped resion at top left
        topcorner= mRgba.submat(0,   height, 0, width);
        ///cover grayscale to BGRA
        Imgproc.cvtColor(intermediate, topcorner, Imgproc.COLOR_GRAY2BGRA, 4);

        ///use this to classify camera feed
        //classifier.classifyMat(CNN_input);
        //Imgproc.putText(mRgba, "Digit: "+classifier.getdigit()+ " Prob: "+classifier.getProb(), new Point(top, left), 3, 3, new Scalar(255, 0, 0, 255), 2);

        graytemp.release();
        topcorner.release();

        return mRgba;
    }

    public void btnOnClick(View view) {
        if(classifier!=null) {
            classifier.classifyMat(CNN_input);
            if(classifier.getCharacter()!='<'){
                result = "Karakter: " + classifier.getCharacter();
                tv.setText(result);
                Toast.makeText(getApplicationContext(),result,Toast.LENGTH_LONG).show();
            }
            else{
                result = "tf modeli yuklenmedi!";
                tv.setText(result);
                Toast.makeText(getApplicationContext(),result,Toast.LENGTH_LONG).show();
            }
        }
        onCameraViewStopped();
        onPause();
    }
}