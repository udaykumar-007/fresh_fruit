package com.example.fruit_app_2;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.Paint;
import ai.onnxruntime.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import android.graphics.Canvas;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.util.concurrent.ExecutorService;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.ai.client.generativeai.GenerativeModel;
import com.google.ai.client.generativeai.java.GenerativeModelFutures;
import com.google.ai.client.generativeai.type.Content;
import com.google.ai.client.generativeai.type.GenerateContentResponse;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import android.widget.SeekBar;
import android.view.View;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final String IMAGE_FILE_NAME = "captured_image.jpg"; // Define your image file name
    private ImageView imageView;
    private Bitmap currentImageBitmap; // To store the captured image for inversion
    private Uri imageUri;
    private SeekBar conf_int_seekbar;
    private float float_conf_lim;
    private final ExecutorService executorService = Executors.newFixedThreadPool(3); // Use a pool to process images
    private final Handler uiHandler = new Handler(Looper.getMainLooper()); // To update the main image on the UI thread
    private GenerativeModel gm;
    private GenerativeModelFutures genaimodel;
    private Bitmap mutableBitmap;
    private ProgressBar progressBar;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.processedImageView);
        Button captureButton = findViewById(R.id.btnCapture);
        Button processButton = findViewById(R.id.btnProcess);
        conf_int_seekbar = findViewById(R.id.conf_int_seekbar_xml);
        progressBar = findViewById(R.id.progressBar);

        try {
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open("opening_img.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            imageView.setImageBitmap(bitmap);
            System.out.println("image set");
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error opening main page image");
        }

        gm = new GenerativeModel(
                /* modelName */ "gemini-1.5-flash-001",
                /* apiKey */ "gemini_api_key");
        genaimodel = GenerativeModelFutures.from(gm);



        captureButton.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
            } else {
                openCamera();


            }
        });

        processButton.setOnClickListener(v -> {
            if (currentImageBitmap != null) {
                try {
                    String modelPath = loadModelFile(this, "yolov8n.onnx");
                    ObjectDetection detector = new ObjectDetection(this, modelPath, float_conf_lim, 0.4f);
                    mutableBitmap = currentImageBitmap.copy(Bitmap.Config.ARGB_8888, true);
                    List<float[]> detections_yolo = detector.runInference(mutableBitmap);

                    System.out.println("Udayy Inferrence done");
                    if(detections_yolo.size() < 10){
                    geminiinf_2(detections_yolo);}
                    else{
                        System.out.println("Huge number of objects detected");
                    }
                    detector.close();

                } catch (OrtException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

            } else {
                Toast.makeText(this, "No image captured to process", Toast.LENGTH_SHORT).show();
            }
        });

        // Set a listener for SeekBar changes
        conf_int_seekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar conf_int_seekbar, int progress, boolean fromUser) {
                // Convert progress to a float between 0 and 1
                float_conf_lim = progress / (float) conf_int_seekbar.getMax();
                // Update TextView with float value
            }

            @Override
            public void onStartTrackingTouch(SeekBar conf_int_seekbar) {
                // Optional: Handle the user starting to touch the SeekBar
            }

            @Override
            public void onStopTrackingTouch(SeekBar conf_int_seekbar) {
                // Optional: Handle the user stopping touching the SeekBar
            }
        });



    }


    public void geminiinf_2(List<float[]> detections){
        progressBar.setVisibility(View.VISIBLE);
        for (float[] detection : detections) {
            executorService.execute(() -> {
                Bitmap cropped_image = get_cropped_image(detection);
                String description=getGeminiOutput(cropped_image);

                System.out.println(description);
                synchronized (mutableBitmap) {
                    drawDetectionsOnImage(mutableBitmap, detection, description);
                }

                uiHandler.post(() -> {
                    imageView.setImageBitmap(mutableBitmap);
                    progressBar.setVisibility(View.GONE);

                });
            });
        }

    }



    public Bitmap get_cropped_image(float[] detection_box){
        float xCenter = detection_box[0];
        float yCenter = detection_box[1];
        float width = detection_box[2];
        float height = detection_box[3];

        float left = (xCenter - width / 2) * ((float) currentImageBitmap.getWidth() / 640);
        float top = (yCenter - height / 2) * ((float) currentImageBitmap.getHeight() / 640);
        float right = (xCenter + width / 2) * ((float) currentImageBitmap.getWidth() / 640);
        float bottom = (yCenter + height / 2) * ((float) currentImageBitmap.getHeight() / 640);

        int classId = (int) detection_box[5];
        float confidence = detection_box[4];
        Bitmap croppedImage = Bitmap.createBitmap(
                currentImageBitmap,
                Math.max(0, (int) left),
                Math.max(0, (int) top),
                Math.min((int) (right - left), currentImageBitmap.getWidth() - (int) left),
                Math.min((int) (bottom - top), currentImageBitmap.getHeight() - (int) top)
        );

        return croppedImage;

    }

    public String getGeminiOutput(Bitmap bitmap_tbinf) {
        Content content =
                new Content.Builder()
                        .addText("Check if the given image has any eatables. If it contains check for the freshness of it, give the output as fruit name and number on scale of 0 to 100 like 'apple:90'. 100 is for good, 0 is for bad. if the image does not contain any eatable give response as 'non-eatable', nothing else. Input images will be objects detected by YOLO model so mostly single object will be present. The eatables will be related to indian foods mostly south indian, give response accordingly")
                        .addImage(bitmap_tbinf)
                        .build();

        Executor executor = Executors.newSingleThreadExecutor();
        CompletableFuture<String> futureResult = new CompletableFuture<>();

        ListenableFuture<GenerateContentResponse> response = genaimodel.generateContent(content);
        Futures.addCallback(
                response,
                new FutureCallback<GenerateContentResponse>() {
                    @Override
                    public void onSuccess(GenerateContentResponse result) {
                        futureResult.complete(result.getText());
                    }

                    @Override
                    public void onFailure(Throwable t) {
                        t.printStackTrace();
                        futureResult.complete("No output");
                    }
                },
                executor
        );

        try {
            // Block until the result is available
            return futureResult.get(); // Blocks until result is available
        } catch (Exception e) {
            e.printStackTrace();
            return "Error in processing";
        }
    }


    private void drawDetectionsOnImage(Bitmap image,float[] detection, String description) {
        Result processed_string = processString(description);
        Canvas canvas = new Canvas(image);
        Paint boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(100);
        Paint fillPaint = new Paint();
        fillPaint.setColor(processed_string.getColor_gemini());
        fillPaint.setStyle(Paint.Style.FILL);

        float xCenter = detection[0];
        float yCenter = detection[1];
        float width = detection[2];
        float height = detection[3];

        float left = (xCenter - width / 2) * ((float) image.getWidth() / 640);
        float top = (yCenter - height / 2) * ((float) image.getHeight() / 640);
        float right = (xCenter + width / 2) * ((float) image.getWidth() / 640);
        float bottom = (yCenter + height / 2) * ((float) image.getHeight() / 640);



        canvas.drawRect(left, top, right, bottom, fillPaint);

        String label;
        if(processed_string.hasNumber()) label = String.format("%s", processed_string.getValue());
        else label = String.format("%s", processed_string.getValue());
        canvas.drawText(label, left, top - 10, textPaint);


    }

    private String loadModelFile(Context context, String modelName) throws IOException {
        File tempModelFile = File.createTempFile("yolov8n", ".onnx", context.getCacheDir());

        try (InputStream is = context.getAssets().open(modelName);
             FileOutputStream fos = new FileOutputStream(tempModelFile)) {
            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }
        }


        return tempModelFile.getAbsolutePath();
    }




    private void openCamera() {
        File imageFile = new File(getExternalFilesDir(null), IMAGE_FILE_NAME);
        imageUri = FileProvider.getUriForFile(this, "com.example.fruit_app_2.fileprovider", imageFile);

        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);

        // Set up additional options for high-resolution photos (optional)
        takePictureIntent.putExtra("android.intent.extras.CAMERA_FACING", 1);  // Use front camera
        takePictureIntent.putExtra("android.intent.extras.SIZE_LIMIT", 10 * 1024 * 1024);  // Limit file size (optional)

        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            try {
                // Load the captured image from the file URI
                currentImageBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                imageView.setImageBitmap(currentImageBitmap);
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Camera permission is required to capture photos", Toast.LENGTH_SHORT).show();
            }
        }
    }

    public static Result processString(String input) {
        String[] keyValue = input.split(":");
        if (keyValue.length == 2) {
            String value = keyValue[1];

            // Match any sequence of digits in the value
            Matcher matcher = Pattern.compile("\\d+").matcher(value);
            if (matcher.find()) {
                int number = Integer.parseInt(matcher.group());
                int color = getColor_gemini(number);
                return new Result(input, true, color);
            }
            return new Result(value, false, Color.argb(100, 128, 128, 128)); // Default to gray if no number found
        }
        return new Result(input, false, Color.argb(100, 128, 128, 128)); // No ":" or no number found
    }

    public static int getColor_gemini(int number) {
        System.out.println(number);
        int alpha = 128; // 50% transparency

        number = Math.max(0, Math.min(100, number));

        // Calculate green and red components
        int green = (int) ((number / 100.0) * 255); // Higher number => more green
        int red = (int) ((1 - (number / 100.0)) * 255); // Lower number => more red

        // Return ARGB color
        return Color.argb(alpha, red, green, 0);
    }

    // Custom Result class
    static class Result {
        private final Object value;
        private final boolean hasNumber;
        private final int color;

        public Result(Object value, boolean hasNumber, int color) {
            this.value = value;
            this.hasNumber = hasNumber;
            this.color = color;
        }

        public Object getValue() {
            return value;
        }

        public boolean hasNumber() {
            return hasNumber;
        }

        public int getColor_gemini() {
            return color;
        }

        @Override
        public String toString() {
            return value + " : " + value +" %";
        }
    }

}
