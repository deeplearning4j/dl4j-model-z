package org.deeplearning4j;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * Deep Big Simple NN
 *
 * Reference: http://arxiv.org/pdf/1003.0358v1.pdf
 */
public class DeepBigSimpleNet {

    // TODO finish reviewing and pulling in paper details
    private int height;
    private int width;
    private int channels;
    private int numLabels;
    private long seed;
    private int iterations;

    public DeepBigSimpleNet(int height, int width, int channels, int numLabels, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerNetwork init() {

        // TODO for mnist example expand training data like Simard et al
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRate(1e-3f) // TODO create learnable lr that shrinks by multiplicative constant after each epoch pg 3
                .updater(Updater.NESTEROVS)
                .momentum(0)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width * channels)
                        .nOut(2500)
                        .activation("tanh") // TODO set A = 1.7159 and B = 0.6666
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(2500)
                        .nOut(2000)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(2000)
                        .nOut(1500)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(1500)
                        .nOut(1000)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(500)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nIn(500)
                        .nOut(numLabels)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }
}