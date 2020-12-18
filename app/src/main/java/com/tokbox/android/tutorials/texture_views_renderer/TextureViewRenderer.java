package com.tokbox.android.tutorials.texture_views_renderer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.media.Image;
import android.opengl.GLES20;
import android.opengl.Matrix;
import android.os.Environment;
import android.util.Log;
import android.view.TextureView;
import android.view.View;

import com.android.grafika.EglCore;
import com.opentok.android.BaseVideoRenderer;
import com.android.grafika.WindowSurface;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.concurrent.locks.ReentrantLock;

class TextureViewRenderer extends BaseVideoRenderer {
    private static final String TAG = TextureViewRenderer.class.getSimpleName();

    private final TextureView view;
    private TextureViewRenderer.Renderer renderer;
    private boolean videoLastStatus;
    private boolean detached = false;
    private boolean mEnableVideoFit = false;
    private Context mContext;

    /**
     * Create a new textureview renderer.
     *
     * @param context App context
     */
    public TextureViewRenderer(Context context) {
        mContext = context;
        view = new TextureView(context);
        renderer = new TextureViewRenderer.Renderer();
        view.setSurfaceTextureListener(renderer);
        View.OnAttachStateChangeListener mStateChangeListener = new View.OnAttachStateChangeListener() {
            @Override
            public void onViewAttachedToWindow(View v) {
                attachView();
            }

            @Override
            public void onViewDetachedFromWindow(View v) {
                detachView();
            }
        };
        view.addOnAttachStateChangeListener(mStateChangeListener);
        renderer.start();
    }

    @Override
    public void onFrame(Frame frame) {
        renderer.displayFrame(frame);
    }

    @Override
    public void setStyle(String key, String value) {
        if (BaseVideoRenderer.STYLE_VIDEO_SCALE.equals(key)) {
            if (BaseVideoRenderer.STYLE_VIDEO_FIT.equals(value)) {
                renderer.enableVideoFit(true);
                mEnableVideoFit = true;
            } else if (BaseVideoRenderer.STYLE_VIDEO_FILL.equals(value)) {
                renderer.enableVideoFit(false);
                mEnableVideoFit = false;
            }
        }
    }

    @Override
    public void onVideoPropertiesChanged(boolean videoEnabled) {
        renderer.setEnableVideo(videoEnabled);
    }

    @Override
    public View getView() {
        return view;
    }

    private void restartRenderer() {
        renderer = new TextureViewRenderer.Renderer();
        renderer.enableVideoFit(mEnableVideoFit);
        view.setSurfaceTextureListener(renderer);
        renderer.start();
    }

    private void attachView() {
        if (detached) {
            detached = false;
            restartRenderer();
        }
    }

    private void detachView() {
        detached = true;
    }

    @Override
    public void onPause() {
        videoLastStatus = renderer.isEnableVideo();
        renderer.setEnableVideo(false);
    }

    @Override
    public void onResume() {
        renderer.setEnableVideo(videoLastStatus);
        if (!renderer.isAlive()) {
            restartRenderer();
        }
    }

    public void takeScreenshot() {
        renderer.takeScreenshot(mContext.getCacheDir() + "/capture.jpeg");
    }

    private static class Renderer extends Thread implements TextureView.SurfaceTextureListener {
        private final String vertexShaderCode = "uniform mat4 uMVPMatrix;"
                + "attribute vec4 aPosition;\n"
                + "attribute vec2 aTextureCoord;\n"
                + "varying vec2 vTextureCoord;\n" + "void main() {\n"
                + "  gl_Position = uMVPMatrix * aPosition;\n"
                + "  vTextureCoord = aTextureCoord;\n" + "}\n";

        private final String fragmentShaderCode = "precision mediump float;\n"
                + "uniform sampler2D Ytex;\n"
                + "uniform sampler2D Utex,Vtex;\n"
                + "varying vec2 vTextureCoord;\n"
                + "void main(void) {\n"
                + "  float nx,ny,r,g,b,y,u,v;\n"
                + "  mediump vec4 txl,ux,vx;"
                + "  nx=vTextureCoord[0];\n"
                + "  ny=vTextureCoord[1];\n"
                + "  y=texture2D(Ytex,vec2(nx,ny)).r;\n"
                + "  u=texture2D(Utex,vec2(nx,ny)).r;\n"
                + "  v=texture2D(Vtex,vec2(nx,ny)).r;\n"
                + "  y=1.1643*(y-0.0625);\n"
                + "  u=u-0.5;\n" + "  v=v-0.5;\n" + "  r=y+1.5958*v;\n"
                + "  g=y-0.39173*u-0.81290*v;\n" + "  b=y+2.017*u;\n"
                + "  gl_FragColor=vec4(r,g,b,1.0);\n" + "}\n";

        final Object lock = new Object();
        EglCore eglCore;
        private boolean videoEnabled = true;
        private boolean videoFitEnabled = false;

        SurfaceTexture surfaceTexture;
        int glProgram;

        final int[] textureIds = new int[3];
        final float[] scaleMatrix = new float[16];

        static final float[] xyzCoords = {
                -1.0f, 1.0f, 0.0f, // top left
                -1.0f, -1.0f, 0.0f, // bottom left
                1.0f, -1.0f, 0.0f, // bottom right
                1.0f, 1.0f, 0.0f // top right
        };

        static final float[] uvCoords = {
                0, 0, // top left
                0, 1, // bottom left
                1, 1, // bottom right
                1, 0 // top right
        };

        private final short[] vertexIndex = {0, 1, 2, 0, 2, 3}; // order to draw
        static final int COORDS_PER_VERTEX = 3;
        static final int TEXTURECOORDS_PER_VERTEX = 2;

        private final FloatBuffer vertexBuffer;
        private final FloatBuffer textureBuffer;
        private final ShortBuffer drawListBuffer;

        private int textureWidth;
        private int textureHeight;
        private int viewportWidth;
        private int viewportHeight;
        private int previousViewPortHeight;
        private int previousViewPortWidth;

        final ReentrantLock frameLock = new ReentrantLock();

        Frame currentFrame;

        private Renderer() {
            ByteBuffer bb = ByteBuffer.allocateDirect(xyzCoords.length * 4);
            bb.order(ByteOrder.nativeOrder());
            vertexBuffer = bb.asFloatBuffer();
            vertexBuffer.put(xyzCoords);
            vertexBuffer.position(0);

            ByteBuffer tb = ByteBuffer.allocateDirect(uvCoords.length * 4);
            tb.order(ByteOrder.nativeOrder());
            textureBuffer = tb.asFloatBuffer();
            textureBuffer.put(uvCoords);
            textureBuffer.position(0);

            ByteBuffer dlb = ByteBuffer.allocateDirect(vertexIndex.length * 2);
            dlb.order(ByteOrder.nativeOrder());
            drawListBuffer = dlb.asShortBuffer();
            drawListBuffer.put(vertexIndex);
            drawListBuffer.position(0);
        }

        @Override
        public void run() {
            waitUntilSurfaceIsReady();

            eglCore = new EglCore(null, EglCore.FLAG_TRY_GLES3);
            WindowSurface windowSurface = new WindowSurface(eglCore, surfaceTexture);
            windowSurface.makeCurrent();
            setupgl();
            renderFrameLoop(windowSurface);

            windowSurface.release();
            eglCore.release();
        }

        // Surface listener imlementation
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture st, int width, int height) {
            synchronized (lock) {
                surfaceTexture = st;
                viewportWidth = width;
                viewportHeight = height;
                lock.notify();
            }
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {
            viewportWidth = width;
            viewportHeight = height;
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            synchronized (lock) {
                surfaceTexture = null;
            }
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            synchronized (lock) {
                surfaceTexture = surface;
            }
        }

        // Interface with outer class
        private void enableVideoFit(boolean videoFit) {
            videoFitEnabled = videoFit;
        }

        private void setEnableVideo(boolean video) {
            videoEnabled = video;
        }

        private boolean isEnableVideo() {
            return videoEnabled;
        }

        private void displayFrame(Frame frame) {
            frameLock.lock();
            if (currentFrame != null) {
                currentFrame.destroy();
            }
            currentFrame = frame;
            frameLock.unlock();
        }

        // Utility methods
        private void waitUntilSurfaceIsReady() {
            synchronized (lock) {
                while (surfaceTexture == null) {
                    try {
                        lock.wait();
                    } catch (InterruptedException ex) {
                        Log.d(TAG, "Waiting for surface ready was interrupted");
                    }
                }
            }
        }

        private void updateViewportSizeIfNeeded() {
            if (previousViewPortWidth != viewportWidth
                    || previousViewPortHeight != viewportHeight) {
                GLES20.glViewport(0, 0, viewportWidth, viewportHeight);
                previousViewPortHeight = viewportHeight;
                previousViewPortWidth = viewportWidth;
            }
        }

        private void initializeTexture(int name, int id, int width, int height) {
            GLES20.glActiveTexture(name);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id);
            GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                    GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
            GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                    GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                    GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                    GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE,
                    width, height, 0, GLES20.GL_LUMINANCE,
                    GLES20.GL_UNSIGNED_BYTE, null);
        }

        private void setupTextures(Frame frame) {
            if (textureIds[0] != 0) {
                GLES20.glDeleteTextures(3, textureIds, 0);
            }
            GLES20.glGenTextures(3, textureIds, 0);

            int width = frame.getWidth();
            int height = frame.getHeight();
            int paddedWidth = (width + 1) >> 1;
            int paddedHeight = (height + 1) >> 1;

            initializeTexture(GLES20.GL_TEXTURE0, textureIds[0], width, height);
            initializeTexture(GLES20.GL_TEXTURE1, textureIds[1], paddedWidth, paddedHeight);
            initializeTexture(GLES20.GL_TEXTURE2, textureIds[2], paddedWidth, paddedHeight);

            textureWidth = frame.getWidth();
            textureHeight = frame.getHeight();
        }

        private void updateTextures(Frame frame) {
            int width = frame.getWidth();
            int height = frame.getHeight();
            int halfWidth = (width + 1) >> 1;
            int halfHeight = (height + 1) >> 1;
            int ySize = width * height;
            int uvSize = halfWidth * halfHeight;

            ByteBuffer bb = frame.getBuffer();
            // If we are reusing this frame, make sure we reset position and
            // limit
            bb.clear();

            if (bb.remaining() == ySize + uvSize * 2) {
                bb.position(0);

                GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1);
                GLES20.glPixelStorei(GLES20.GL_PACK_ALIGNMENT, 1);

                GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureIds[0]);
                GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, width,
                        height, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE,
                        bb);

                bb.position(ySize);
                GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureIds[1]);
                GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0,
                        halfWidth, halfHeight, GLES20.GL_LUMINANCE,
                        GLES20.GL_UNSIGNED_BYTE, bb);

                bb.position(ySize + uvSize);
                GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureIds[2]);
                GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0,
                        halfWidth, halfHeight, GLES20.GL_LUMINANCE,
                        GLES20.GL_UNSIGNED_BYTE, bb);
            } else {
                textureWidth = 0;
                textureHeight = 0;
            }

        }

        private void renderFrameLoop(WindowSurface surface) {
            while (true) {
                synchronized (lock) {
                    if (surfaceTexture == null) {
                        break;
                    }
                }

                frameLock.lock();
                if (currentFrame != null && videoEnabled) {
                    GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
                    GLES20.glUseProgram(glProgram);

                    updateViewportSizeIfNeeded();

                    if (textureWidth != currentFrame.getWidth()
                            || textureHeight != currentFrame.getHeight()) {
                        setupTextures(currentFrame);
                    }
                    updateTextures(currentFrame);

                    Matrix.setIdentityM(scaleMatrix, 0);
                    float scalex = 1.0f;
                    float scaley = 1.0f;
                    float ratio = (float) currentFrame.getWidth()
                            / currentFrame.getHeight();
                    float vratio = (float) viewportWidth / viewportHeight;

                    if (videoFitEnabled) {
                        if (ratio > vratio) {
                            scaley = vratio / ratio;
                        } else {
                            scalex = ratio / vratio;
                        }
                    } else {
                        if (ratio < vratio) {
                            scaley = vratio / ratio;
                        } else {
                            scalex = ratio / vratio;
                        }
                    }

                    Matrix.scaleM(scaleMatrix, 0,
                            scalex * (currentFrame.isMirroredX() ? -1.0f : 1.0f),
                            scaley, 1);

                    int mvpMatrix = GLES20.glGetUniformLocation(glProgram,
                            "uMVPMatrix");
                    GLES20.glUniformMatrix4fv(mvpMatrix, 1, false,
                            scaleMatrix, 0);

                    GLES20.glDrawElements(GLES20.GL_TRIANGLES, vertexIndex.length,
                            GLES20.GL_UNSIGNED_SHORT, drawListBuffer);
                } else {
                    //black frame when video is disabled
                    GLES20.glClearColor(0, 0, 0, 1);
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
                }
                frameLock.unlock();
                surface.swapBuffers();
            }
        }

        private int loadShader(int type, String shaderCode) {
            int shader = GLES20.glCreateShader(type);

            GLES20.glShaderSource(shader, shaderCode);
            GLES20.glCompileShader(shader);

            return shader;
        }

        private void setupgl() {
            GLES20.glClearColor(0, 0, 0, 1);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

            int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER,
                    vertexShaderCode);
            int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER,
                    fragmentShaderCode);

            glProgram = GLES20.glCreateProgram(); // create empty OpenGL ES
            // Program
            GLES20.glAttachShader(glProgram, vertexShader); // add the vertex
            // shader to program
            GLES20.glAttachShader(glProgram, fragmentShader); // add the fragment
            // shader to
            // program
            GLES20.glLinkProgram(glProgram);

            int positionHandle = GLES20.glGetAttribLocation(glProgram,
                    "aPosition");
            int textureHandle = GLES20.glGetAttribLocation(glProgram,
                    "aTextureCoord");

            GLES20.glVertexAttribPointer(positionHandle, COORDS_PER_VERTEX,
                    GLES20.GL_FLOAT, false, COORDS_PER_VERTEX * 4,
                    vertexBuffer);

            GLES20.glEnableVertexAttribArray(positionHandle);

            GLES20.glVertexAttribPointer(textureHandle,
                    TEXTURECOORDS_PER_VERTEX, GLES20.GL_FLOAT, false,
                    TEXTURECOORDS_PER_VERTEX * 4, textureBuffer);

            GLES20.glEnableVertexAttribArray(textureHandle);

            GLES20.glUseProgram(glProgram);
            int identifier = GLES20.glGetUniformLocation(glProgram, "Ytex");
            GLES20.glUniform1i(identifier, 0); /* Bind Ytex to texture unit 0 */

            identifier = GLES20.glGetUniformLocation(glProgram, "Utex");
            GLES20.glUniform1i(identifier, 1); /* Bind Utex to texture unit 1 */

            identifier = GLES20.glGetUniformLocation(glProgram, "Vtex");
            GLES20.glUniform1i(identifier, 2); /* Bind Vtex to texture unit 2 */

            textureWidth = 0;
            textureHeight = 0;
        }

        public void takeScreenshot(String path) {
            frameLock.lock();
            currentFrame.getBuffer().rewind();
            byte[] framebuffer = new byte[currentFrame.getBuffer().remaining()];
            currentFrame.getBuffer().get(framebuffer);
            int[] strides = { currentFrame.getYstride(), currentFrame.getUvStride() };
            int h = currentFrame.getHeight();
            int w = currentFrame.getWidth();
            frameLock.unlock();

            Bitmap b = toRGB(framebuffer, w, h, strides[0], strides[1]);
            Log.d(TAG, "Saving capture to: " + path);
            try (FileOutputStream f = new FileOutputStream(path)) {
                b.compress(Bitmap.CompressFormat.JPEG, 100, f);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public Bitmap toRGB(byte[] yuvBuffer, int w, int h, int yStride, int uvStride) {
            int[] rgbBytes = new int[h*w];
            int idx = 0;
            int uvBufferStart =  w * h;
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    int y = yuvBuffer[row * yStride + col] & 0xff;
                    int u = yuvBuffer[uvBufferStart + (row/2 * uvStride) + col/2] & 0xff;
                    int v = yuvBuffer[uvBufferStart + (row/2 * uvStride) + col/2] & 0xff;

                    int y1 = ((19077 << 8) * y) >> 16;
                    int r = (y1 + (((26149 << 8) * v) >> 16) - 14234) >> 6;
                    int g = (y1 - (((6419 << 8) * u) >> 16) - (((13320 << 8) * v) >> 16) +  8708) >> 6;
                    int b = (y1 + (((33050 << 8) * u) >> 16) - 17685) >> 6;

                    if (r < 0) r = 0;
                    if (g < 0) g = 0;
                    if (b < 0) b = 0;
                    if (r > 255) r = 255;
                    if (g > 255) g = 255;
                    if (b > 255) b = 255;
                    rgbBytes[idx++] = 0xff000000 + b + 256 * (g + 256 * r);
                }
            }
            return Bitmap.createBitmap(rgbBytes, w, h, Bitmap.Config.ARGB_8888);
        }
    }
}
