package org.deeplearning4j;

import org.deeplearning4j.module.Inception;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.graph.L2Vertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * FaceNetVariant
 *  Reference: https://arxiv.org/abs/1503.03832
 *  Also based on the OpenFace implementation: http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf
 *
 *  Revised and consolidated version by @crockpotveggies
 *
 * Warning this has not been run yet.
 * There are a couple known issues with CompGraph regarding combining different layer types into one and
 * combining different shapes of input even if the layer types are the same at least for CNN.
 */

public class FaceNetVariant {

    private int height;
    private int width;
    private int channels = 3;
    private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 90;

    public FaceNetVariant(int height, int width, int channels, int outputNum, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    public ComputationGraph init() {

        GraphBuilder graph = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation("relu")
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .biasLearningRate(2 * 1e-2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.96)
            .lrPolicySteps(320000)
            .updater(Updater.NESTEROVS)
            .momentum(0.9)
            .weightInit(WeightInit.XAVIER)
            .regularization(true)
            .l2(2e-4)
            .graphBuilder();


        graph
            .addInputs("input1","input2","input3")
            .addVertex("stack1", new StackVertex(), "input1","input2","input3")
            .addLayer("cnn1", Inception.conv7x7(this.channels, 64, 0.2), "stack1")
            .addLayer("max1", new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}, new int[]{0,0}).build(), "cnn1")
            .addLayer("lrn1", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "max1")
            .addLayer("cnn3", Inception.conv3x3(64, 192, 0.2), "cnn2")
            .addLayer("lrn2", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "cnn3")
            .addLayer("max2", new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}, new int[]{0,0}).build(), "lrn2");

        Inception.updateBuilder(graph, "3a", 192, new int[][]{{64},{96, 128},{16, 32}, {32}}, "max2");
        Inception.updateBuilder(graph, "3b", 256, new int[][]{{128},{128, 192},{32, 96}, {64}}, "3a-depthconcat1");

        graph.addLayer("max3", new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}, new int[]{0,0}).build(), "3b-depthconcat1");

        Inception.updateBuilder(graph, "4a", 480, new int[][]{{192},{96, 208},{16, 48}, {64}}, "3b-depthconcat1");
        Inception.updateBuilder(graph, "4b", 512, new int[][]{{160},{112, 224},{24, 64}, {64}}, "4a-depthconcat1");

        Inception.updateBuilder(graph, "4c", 512, new int[][]{{128},{128, 256},{24, 64}, {64}}, "4b-depthconcat1");
        Inception.updateBuilder(graph, "4d", 512, new int[][]{{112},{144, 288},{32, 64}, {64}}, "4c-depthconcat1");

        Inception.updateBuilder(graph, "4e", 528, new int[][]{{256},{160, 320},{32, 128}, {128}}, "4d-depthconcat1");

        graph.addLayer("max4", new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}, new int[]{0,0}).build(), "4e-depthconcat1");

        Inception.updateBuilder(graph, "5a", 832, new int[][]{{256},{160, 320},{32, 128}, {128}}, "max4");
        Inception.updateBuilder(graph, "5b", 832, new int[][]{{384},{192, 384},{48, 128}, {128}}, "5a-depthconcat1");

        graph.addLayer("avg3", Inception.avgPool7x7(1), "5b-depthconcat1") // output: 1x1x1024
            .addLayer("fc1", Inception.fullyConnected(1024, 1024, 0.4), "avg3") // output: 1x1x1024
            .addVertex("unstack0", new UnstackVertex(0), "fc1")
            .addVertex("unstack1", new UnstackVertex(1), "fc1")
            .addVertex("unstack2", new UnstackVertex(2), "fc1")
            .addVertex("l2-1", new L2Vertex(), "unstack1", "unstack0") // x - x-
            .addVertex("l2-2", new L2Vertex(), "unstack1", "unstack2") // x - x+
            .addLayer("lossLayer", new LossLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation("softmax")
                .build(), "l2-1", "l2-2")
            .setOutputs("lossLayer")
            .backprop(true).pretrain(false);

        ComputationGraphConfiguration conf = graph.build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }



}
