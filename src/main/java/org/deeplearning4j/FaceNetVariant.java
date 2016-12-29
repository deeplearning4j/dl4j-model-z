package org.deeplearning4j;

import org.deeplearning4j.module.Inception;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.L2Vertex;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * A variant of the original FaceNet model that relies on embeddings and triplet loss.
 * Reference: https://arxiv.org/abs/1503.03832
 * Also based on the OpenFace implementation: http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf
 *
 * Revised and consolidated version by @crockpotveggies
 */

public class FaceNetVariant {

    private int height;
    private int width;
    private int channels = 3;
    //  private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 1;
    private boolean miniBatch = true;
    private double learningRate = 0.001;
    private String activation = "relu";
    private WeightInit weightInit = WeightInit.RELU;

    private double adamVarDecay = 0.999;
    private double adamMeanDecay = 0.9;

    public FaceNetVariant(int height, int width, int channels, long seed, int iterations, boolean miniBatch,
                          double learningRate, String activation, WeightInit weightInit) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.seed = seed;
        this.iterations = iterations;
        this.miniBatch = miniBatch;
        this.learningRate = learningRate;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    public ComputationGraph init() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation(activation)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .updater(Updater.ADAM)
            .adamVarDecay(adamVarDecay)
            .adamMeanDecay(adamMeanDecay)
            .weightInit(weightInit)
            .regularization(false)
//        .l1(0.01)
//        .l2(0.001)
//        .dropOut(0.999)
            .learningRate(learningRate)
//        .biasLearningRate(1e-2*2)
            .miniBatch(miniBatch)
            .convolutionMode(ConvolutionMode.Same)
            .graphBuilder();


        graph
            .addInputs("input1","input2","input3")
            .setInputTypes(InputType.convolutional(height, width, channels),InputType.convolutional(height, width, channels),InputType.convolutional(height, width, channels))
            .addVertex("stack1", new StackVertex(), "input1","input2","input3")
            .addLayer("cnn1", Inception.conv7x7(this.channels, 64, 0.2), "stack1")
            .addLayer("batch1", new BatchNormalization.Builder(1e-4, 0.75).nIn(64).nOut(64).build(), "cnn1")

            // pool -> norm
            .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}, new int[]{1,1}).build(), "batch1")
            .addLayer("lrn1", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "pool1")

            // Inception 2
            .addLayer("inception-2-cnn1", Inception.conv1x1(64, 64, 0.2), "lrn1")
            .addLayer("inception-2-batch1", new BatchNormalization.Builder(false).nIn(64).nOut(64).build(), "inception-2-cnn1")
            .addLayer("inception-2-cnn2", Inception.conv3x3(64, 192, 0.2), "inception-2-batch1")
            .addLayer("inception-2-batch2", new BatchNormalization.Builder(false).nIn(192).nOut(192).build(), "inception-2-cnn2")

            // norm -> pool
            .addLayer("inception-2-lrn1", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "inception-2-batch2")
            .addLayer("inception-2-pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}, new int[]{1,1}).build(), "inception-2-lrn1");

        // Inception 3a
        Inception.appendGraph(graph, "3a", 192,
            new int[]{3,5}, new int[]{1,1}, new int[]{128,32}, new int[]{96,16,32,64},
            SubsamplingLayer.PoolingType.MAX, true, "inception-2-pool1");
        // Inception 3b
        Inception.appendGraph(graph, "3b", 256,
            new int[]{3,5}, new int[]{1,1}, new int[]{128,64}, new int[]{96,32,64,64},
            SubsamplingLayer.PoolingType.PNORM, 2, true, "inception-3a");
        Inception.appendGraph(graph, "3c", 320,
            new int[]{3,5}, new int[]{2,2}, new int[]{256,64}, new int[]{128,32},
            SubsamplingLayer.PoolingType.MAX, true, "inception-3b");

        // Inception 4a
        Inception.appendGraph(graph, "4a", 320,
            new int[]{3,5}, new int[]{1,1}, new int[]{192,64}, new int[]{96,32,128,256},
            SubsamplingLayer.PoolingType.PNORM, 2, true, "inception-3c");
        // Inception 4e
        Inception.appendGraph(graph, "4e", 640,
            new int[]{3,5}, new int[]{2,2}, new int[]{256,128}, new int[]{160,64},
            SubsamplingLayer.PoolingType.MAX, 2, 1, true, "inception-4a");

        // Inception 5a
        Inception.appendGraph(graph, "5a", 384,
            new int[]{3}, new int[]{1}, new int[]{384}, new int[]{96,96,256},
            SubsamplingLayer.PoolingType.PNORM, 2, true, "inception-4e");
        // Inception 5b
        Inception.appendGraph(graph, "5b", 736,
            new int[]{3}, new int[]{1}, new int[]{384}, new int[]{96,96,256},
            SubsamplingLayer.PoolingType.MAX, 1, 1, true, "inception-5a");

        graph
            .addLayer("avg3", Inception.avgPoolNxN(3,3), "inception-5b") // output: 1x1x1024
            .addLayer("embed1", new DenseLayer.Builder().nIn(736).nOut(128).activation("identity").build(), "avg3")
            .addVertex("unstack0", new UnstackVertex(0,3), "embed1")
            .addVertex("unstack1", new UnstackVertex(1,3), "embed1")
            .addVertex("unstack2", new UnstackVertex(2,3), "embed1")
            .addVertex("l2-1", new L2Vertex(), "unstack1", "unstack0") // x - x-
            .addVertex("l2-2", new L2Vertex(), "unstack1", "unstack2") // x - x+
            .addLayer("lossLayer", new LossLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .build(), "l2-1", "l2-2")
            .setOutputs("lossLayer")
            .backprop(true).pretrain(false);

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        System.out.println("\nNumber of params: "+model.numParams()+"\n");

        return model;
    }

}