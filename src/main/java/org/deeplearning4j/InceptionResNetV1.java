package org.deeplearning4j;

import org.deeplearning4j.module.InceptionResNet;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Inception-ResNet combined residual shortcuts in neural network models
 * with the GoogleLeNet Inception configurations that maximize efficiency
 * in computing.
 *
 * This is version 1 from the Inception-ResNet paper: https://arxiv.org/abs/1602.07261
 *
 * Revised and consolidated. Likely needs further tuning for specific applications.
 *
 * @author Justin Long (crockpotveggies)
 */
public class InceptionResNetV1 {

    private int height;
    private int width;
    private int channels = 3;
    private long seed = 123;
    private int iterations = 90;
    private boolean miniBatch = true;
    private double learningRate = 0.001;
    private Activation activation = Activation.RELU;
    private double epsilon = 0.1;

    private double adamVarDecay = 0.999;
    private double adamMeanDecay = 0.9;

    public InceptionResNetV1(int height, int width, int channels, long seed, int iterations,
                      boolean miniBatch, double learningRate) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.seed = seed;
        this.iterations = iterations;
        this.miniBatch = miniBatch;
        this.learningRate = learningRate;
    }

    public ComputationGraph initTraining(int numClasses) {
        int embeddingSize = 128;
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder("input1", embeddingSize);

        graph
            .addInputs("input1")
            .setInputTypes(InputType.convolutional(height, width, channels))
            .addLayer("outputLayer", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(embeddingSize).nOut(numClasses)
                .build(), "embeddings")
            .setOutputs("outputLayer")
            .backprop(true).pretrain(false);

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        System.out.println("\nNumber of params: "+model.numParams()+"\n");

        return model;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input, int embeddingSize) {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation(activation)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.ADAM)
            .adamVarDecay(adamVarDecay)
            .adamMeanDecay(adamMeanDecay)
            .epsilon(epsilon)
            .weightInit(WeightInit.XAVIER)
            .regularization(true)
            .l2(2e-4)
            .dropOut(0.8)
            .learningRate(learningRate)
            .miniBatch(miniBatch)
            .convolutionMode(ConvolutionMode.Truncate)
            .graphBuilder();


        graph
            // stem
            .addLayer("stem-cnn1", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(channels).nOut(32).build(), input)
            .addLayer("stem-cnn2", new ConvolutionLayer.Builder(new int[]{3,3}).nIn(32).nOut(32).build(), "stem-cnn1")
            .addLayer("stem-cnn3", new ConvolutionLayer.Builder(new int[]{3,3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(64).build(), "stem-cnn2")
            .addLayer("stem-pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}).build(), "stem-cnn3")
            .addLayer("stem-cnn5", new ConvolutionLayer.Builder(new int[]{1,1}).nIn(64).nOut(80).build(), "stem-pool4")
            .addLayer("stem-cnn6", new ConvolutionLayer.Builder(new int[]{3,3}).nIn(80).nOut(192).build(), "stem-cnn5")
            .addLayer("stem-cnn7", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(192).nOut(256).build(), "stem-cnn6");


        // 5xInception-resnet-A
        InceptionResNet.inceptionV1ResA(graph, "resnetA", 5, "stem-cnn7");


        // Reduction-A
        graph
            // 3x3
            .addLayer("reduceA-cnn1", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(256).nOut(256).build(), "resnetA")
            // 1x1 -> 3x3 -> 3x3
            .addLayer("reduceA-cnn2", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(192).build(), "resnetA")
            .addLayer("reduceA-cnn3", new ConvolutionLayer.Builder(new int[]{3,3}).convolutionMode(ConvolutionMode.Same).nIn(192).nOut(192).build(), "reduceA-cnn2")
            .addLayer("reduceA-cnn4", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(192).nOut(256).build(), "reduceA-cnn3")
            // maxpool
            .addLayer("reduceA-pool5", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}).build(), "resnetA")
            // -->
            .addVertex("reduceA", new MergeVertex(), "reduceA-cnn1", "reduceA-cnn4", "reduceA-pool5");


        // 10xInception-resnet-B
        InceptionResNet.inceptionV1ResB(graph, "resnetB", 10, "reduceA");


        // Reduction-B
        graph
            // 3x3 pool
            .addLayer("reduceB-pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}).build(), "resnetB")
            // 1x1 -> 3x3
            .addLayer("reduceB-cnn2", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(768).nOut(256).build(), "resnetB")
            .addLayer("reduceB-cnn3", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(256).nOut(256).build(), "reduceB-cnn2")
            // 1x1 -> 3x3
            .addLayer("reduceB-cnn4", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(768).nOut(256).build(), "resnetB")
            .addLayer("reduceB-cnn5", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(256).nOut(256).build(), "reduceB-cnn4")
            // 1x1 -> 3x3 -> 3x3
            .addLayer("reduceB-cnn6", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(768).nOut(256).build(), "resnetB")
            .addLayer("reduceB-cnn7", new ConvolutionLayer.Builder(new int[]{3,3}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(256).build(), "reduceB-cnn6")
            .addLayer("reduceB-cnn8", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(256).nOut(256).build(), "reduceB-cnn7")
            // -->
            .addVertex("reduceB", new MergeVertex(), "reduceB-pool1", "reduceB-cnn3", "reduceB-cnn5", "reduceB-cnn8");


        // 10xInception-resnet-C
        InceptionResNet.inceptionV1ResC(graph, "resnetC", 5, "reduceB");

        // Average pooling
        graph.addLayer("averagepool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3,3}).build(), "resnetC");

        // Embeddings
        graph.addLayer("embeddings", new DenseLayer.Builder().nIn(1536).nOut(embeddingSize).activation(Activation.IDENTITY).build(), "averagepool1");

        return graph;
    }

    /**
     * Check how many parameters exist.
     * @param args
     * @throws Exception
     */
    public static void main(String... args) throws Exception {
        ComputationGraph faceNet = new InceptionResNetV1(160, 160, 3, 42, 1, true, 0.003).initTraining(128);
        System.out.println("Number of parameters: "+faceNet.numParams());
    }

}