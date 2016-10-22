package org.deeplearning4j;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Architecture partially based on DeepFace: http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf
 *
 * Used for problems like LFW
 */
public class DeepFaceVariant {

    private int height;
    private int width;
    private int channels;
    private int numLabels;
    private long seed;
    private int iterations;

    public DeepFaceVariant(int height, int width, int channels, int numLabels, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerNetwork init() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .l2(1e-3)
                .updater(Updater.ADAGRAD)
                .useDropConnect(true)
                .list()
                .layer(0, new ConvolutionLayer.Builder(4, 4)
                        .name("cnn1")
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1,1)
                        .nOut(40)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(60)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .build())
                .layer(6, new ConvolutionLayer.Builder(2, 2)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(80)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(160)
                        .dropOut(0.5)
                        .build())
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }
}