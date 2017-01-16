package org.deeplearning4j.module;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;

/**
 * Inception is based on GoogleLeNet configuration of convolutional layers for optimization of
 * resources and learning. You can use this module to add Inception to your own custom models.
 *
 * The GoogleLeNet paper: https://arxiv.org/abs/1409.4842
 *
 * This module is based on the Inception-ResNet paper that combined residual shortcuts with
 * Inception-style networks: https://arxiv.org/abs/1602.07261
 *
 * Revised and consolidated. Likely needs further tuning for specific applications.
 *
 * @author Justin Long (crockpotveggies)
 */
public class InceptionResNet {

    public static String nameLayer(String blockName, String layerName, int i) { return blockName+"-"+layerName+"-"+i; }

    /**
     * Append Inception-ResNet A to a computation graph.
     * @param graph
     * @param blockName
     * @param scale
     * @param input
     * @return
     */
    public static ComputationGraphConfiguration.GraphBuilder inceptionV1ResA(ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale, String input) {
        // first add the RELU activation layer
        graph.addLayer(nameLayer(blockName,"activation1",0), new ActivationLayer.Builder().activation(Activation.RELU).build(), input);

        // loop and add each subsequent resnet blocks
        String previousBlock = nameLayer(blockName,"activation1",0);
        for(int i=1; i<=scale; i++) {
            graph
                // 1x1
                .addLayer(nameLayer(blockName,"cnn1",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                // 1x1 -> 3x3
                .addLayer(nameLayer(blockName,"cnn2",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                .addLayer(nameLayer(blockName,"cnn3",i), new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn2",i))
                // 1x1 -> 3x3 -> 3x3
                .addLayer(nameLayer(blockName,"cnn4",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                .addLayer(nameLayer(blockName,"cnn5",i), new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn4",i))
                .addLayer(nameLayer(blockName,"cnn6",i), new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn5",i))
                // --> 1x1 -->
                .addVertex(nameLayer(blockName,"merge1",i), new MergeVertex(), nameLayer(blockName,"cnn1",i), nameLayer(blockName,"cnn3",i), nameLayer(blockName,"cnn6",i))
                .addLayer(nameLayer(blockName,"cnn7",i), new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(96).nOut(256).build(), nameLayer(blockName,"merge1",i))
                // -->
                .addLayer(nameLayer(blockName,"shortcut-identity",i), new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), previousBlock)
                .addVertex(nameLayer(blockName,"shortcut",i), new ElementWiseVertex(ElementWiseVertex.Op.Add), nameLayer(blockName,"cnn7",i), nameLayer(blockName,"shortcut-identity",i));

            previousBlock = nameLayer(blockName,"shortcut",i);
        }

        graph.addLayer(blockName, new ActivationLayer.Builder().activation(Activation.RELU).build(), previousBlock);
        return graph;
    }

    /**
     * Append Inception-ResNet B to a computation graph.
     * @param graph
     * @param blockName
     * @param scale
     * @param input
     * @return
     */
    public static ComputationGraphConfiguration.GraphBuilder inceptionV1ResB(ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale, String input) {
        // first add the RELU activation layer
        graph.addLayer(nameLayer(blockName,"activation1",0), new ActivationLayer.Builder().activation(Activation.RELU).build(), input);

        // loop and add each subsequent resnet blocks
        String previousBlock = nameLayer(blockName,"activation1",0);
        for(int i=1; i<=scale; i++) {
            graph
                // 1x1
                .addLayer(nameLayer(blockName,"cnn1",i), new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(640).nOut(128).build(), previousBlock)
                // 1x1 -> 3x3 -> 3x3
                .addLayer(nameLayer(blockName,"cnn2",i), new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(640).nOut(128).build(), previousBlock)
                .addLayer(nameLayer(blockName,"cnn3",i), new ConvolutionLayer.Builder(new int[]{1,7}).convolutionMode(ConvolutionMode.Same).nIn(128).nOut(128).build(), nameLayer(blockName,"cnn2",i))
                .addLayer(nameLayer(blockName,"cnn4",i), new ConvolutionLayer.Builder(new int[]{7,1}).convolutionMode(ConvolutionMode.Same).nIn(128).nOut(128).build(), nameLayer(blockName,"cnn3",i))
                // --> 1x1 -->
                .addVertex(nameLayer(blockName,"merge1",i), new MergeVertex(), nameLayer(blockName,"cnn1",i), nameLayer(blockName,"cnn4",i))
                .addLayer(nameLayer(blockName,"cnn5",i), new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(640).nOut(640).build(), previousBlock)
                // -->
                .addLayer(nameLayer(blockName,"shortcut-identity",i), new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), previousBlock)
                .addVertex(nameLayer(blockName,"shortcut",i), new ElementWiseVertex(ElementWiseVertex.Op.Add), nameLayer(blockName,"cnn5",i), nameLayer(blockName,"shortcut-identity",i));

            previousBlock = nameLayer(blockName,"shortcut",i);
        }

        graph.addLayer(blockName, new ActivationLayer.Builder().activation(Activation.RELU).build(), previousBlock);
        return graph;
    }

    /**
     * Append Inception-ResNet C to a computation graph.
     * @param graph
     * @param blockName
     * @param scale
     * @param input
     * @return
     */
    public static ComputationGraphConfiguration.GraphBuilder inceptionV1ResC(ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale, String input) {
        // first add the RELU activation layer
        graph.addLayer(nameLayer(blockName,"activation1",0), new ActivationLayer.Builder().activation(Activation.RELU).build(), input);

        // loop and add each subsequent resnet blocks
        String previousBlock = nameLayer(blockName,"activation1",0);
        for(int i=1; i<=scale; i++) {
            graph
                // 1x1
                .addLayer(nameLayer(blockName,"cnn1",i), new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(1408).nOut(192).build(), previousBlock)
                // 1x1 -> 1x3 -> 3x1
                .addLayer(nameLayer(blockName,"cnn2",i), new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(1408).nOut(192).build(), previousBlock)
                .addLayer(nameLayer(blockName,"cnn3",i), new ConvolutionLayer.Builder(new int[]{1,3}).convolutionMode(ConvolutionMode.Same).nIn(192).nOut(192).build(), nameLayer(blockName,"cnn2",i))
                .addLayer(nameLayer(blockName,"cnn4",i), new ConvolutionLayer.Builder(new int[]{3,1}).convolutionMode(ConvolutionMode.Same).nIn(192).nOut(192).build(), nameLayer(blockName,"cnn3",i))
                // --> 1x1 -->
                .addVertex(nameLayer(blockName,"merge1",i), new MergeVertex(), nameLayer(blockName,"cnn1",i), nameLayer(blockName,"cnn4",i))
                .addLayer(nameLayer(blockName,"cnn5",i), new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(1408).nOut(1408).build(), previousBlock)
                // -->
                .addLayer(nameLayer(blockName,"shortcut-identity",i), new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), previousBlock)
                .addVertex(nameLayer(blockName,"shortcut",i), new ElementWiseVertex(ElementWiseVertex.Op.Add), nameLayer(blockName,"cnn5",i), nameLayer(blockName,"shortcut-identity",i));

            previousBlock = nameLayer(blockName,"shortcut",i);
        }

        graph.addLayer(blockName, new ActivationLayer.Builder().activation(Activation.RELU).build(), previousBlock);
        return graph;
    }

}