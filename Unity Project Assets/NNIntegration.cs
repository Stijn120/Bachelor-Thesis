using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class NNIntegration : MonoBehaviour
{
    private Texture2D tx2D;
    private Model RuntimeModel;

    public NNModel NeuralNetwork;
    public Texture2D TestImage;
    public GameObject ImageScreen;

    public Camera cam;

    public bool processImages;

    public int ComplexityLevel;

    // Start is called before the first frame update
    void Start()
    {
        int n = 0;
        var HeadPoseImages = System.IO.Directory.GetFiles("C:/Users/stijn/OneDrive/Studie/Bachelor Thesis/Neural Network/SelectedHeadPoseImages/");
        var dInfo = new DirectoryInfo("C:/Users/stijn/OneDrive/Studie/Bachelor Thesis/Neural Network/SelectedHeadPoseImages/");
        var FileNames = dInfo.GetFiles();

        if (processImages == true)
        {
            n = HeadPoseImages.Length;
        }
        else
        {
            n = 1;
        }

        for (int i = 0; i < n; i++)
        {
            tx2D = TestImage;
            var imgName = TestImage.name;
            if (processImages == true)
            {
                var bytes = File.ReadAllBytes(HeadPoseImages[i]);
                Debug.Log(bytes.Length);
                tx2D.LoadImage(bytes);
                imgName = FileNames[i].Name;
            }

            RuntimeModel = ModelLoader.Load(NeuralNetwork);

            tx2D = TestImage;

            ImageScreen.GetComponent<RawImage>().texture = tx2D;

            bool verbose = false;
            var worker = WorkerFactory.CreateWorker(RuntimeModel, WorkerFactory.Device.GPU, verbose);

            tx2D = Resize(tx2D, 80, 80);
            var Input = new Tensor(tx2D, 1);

            worker.Execute(Input);
            var output = worker.PeekOutput();

            float[] values = output.AsFloats();
            var prediction = Argmax(values);
            ApplyOutput(prediction);
            Debug.Log(prediction);

            RenderTexture renderTexture = new RenderTexture(504, 504, 1);
            cam.targetTexture = renderTexture;
            cam.Render();
            RenderTexture.active = renderTexture;

            Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
            image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
            image.Apply();

            ToBinary(image, 63, imgName, prediction);

            cam.targetTexture = null;
            RenderTexture.active = null;
            Destroy(renderTexture);

            output.Dispose();
            worker.Dispose();
            Input.Dispose();


        }

    }

    void ToBinary(Texture2D source, int resolution, string imgName, int prediction)
    {
        var inputPixels = source.GetPixels();
        var pixels = new int[resolution * resolution];
        var p = 0;
        int stepSize = 504 / resolution;

        for (int xrec = 0; xrec < 504; xrec += stepSize)
        {
            for (int yrec = 0; yrec < 504; yrec += stepSize)
            {
                pixels[p] = GetReceptiveField(source, xrec, yrec, stepSize);
                p++;
            }
        }

        var sum = 0;
        for (int i = 0; i < resolution * resolution; i++)
        {
            sum += pixels[i];
        }
        var avg = sum / pixels.Length;
        Debug.Log(avg);

        StreamWriter writer = new StreamWriter("C:/Users/stijn/OneDrive/Studie/Bachelor Thesis/Neural Network/ModelOutputImages/Complexity " + ComplexityLevel + "/" + imgName + "_label=" + prediction + ".txt");
        for (int i = resolution - 1; i >= 0; i--)
        {
            for (int j = 0; j < resolution; j++)
            {

                if (pixels[j * resolution + i] > avg)
                {
                    writer.Write(1 + " ");
                }
                else
                {
                    writer.Write(0 + " ");
                }

            }


            //if(counter%63 == 0)
            //{
            writer.Write('\n');

        }

        writer.Close();
    }

    int GetReceptiveField(Texture2D input, int xrec, int yrec, int stepSize)
    {
        int counter = 0;

        for (int x = 0; x < stepSize; x++)
        {

            for (int y = 0; y < stepSize; y++)
            {
                var pixel = input.GetPixel(xrec + x, yrec + y);
                if (!pixel.Equals(new Color(0, 0, 0, 0)))
                {
                    counter++;
                }

            }

        }

        return counter;
    }

    // Update is called once per frame
    void Update()
    {

    }

    // Code of this function is based on https://towardsdatascience.com/how-to-build-your-tensorflow-keras-model-into-an-augmented-reality-app-18405c36acf5
    Texture2D Resize(Texture2D source, int newWidth, int newHeight)
    {
        source.filterMode = FilterMode.Point;
        RenderTexture rt = RenderTexture.GetTemporary(newWidth, newHeight);
        rt.filterMode = FilterMode.Point;
        RenderTexture.active = rt;
        Graphics.Blit(source, rt);
        Texture2D nTex = new Texture2D(newWidth, newHeight);
        nTex.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
        nTex.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        return nTex;
    }

    int Argmax(float[] numbers)
    {
        int maxIndex = 0;

        for (int i = 0; i < numbers.Length; i++)
        {
            if (numbers[maxIndex] < numbers[i])
            {
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    void ApplyOutput(int prediction)
    {
        float DeltaX = 0;
        float DeltaY = 0;
        float DeltaZ = 0;

        switch (prediction)
        {
            case 0:
                break;
            case 1:
                DeltaY = 90;
                break;
            case 2:
                DeltaY = -90;
                break;
            case 3:
                DeltaX = 60;
                break;
            case 4:
                DeltaX = -60;
                break;
            case 5:
                DeltaX = 60;
                DeltaY = -90;
                break;
            case 6:
                DeltaX = 60;
                DeltaY = 90;
                break;
            case 7:
                DeltaX = -60;
                DeltaY = -90;
                break;
            case 8:
                DeltaX = -60;
                DeltaY = 90;
                break;
        }

        gameObject.transform.eulerAngles = new Vector3(
            DeltaX,
            DeltaY,
            DeltaZ
        );
    }

}
