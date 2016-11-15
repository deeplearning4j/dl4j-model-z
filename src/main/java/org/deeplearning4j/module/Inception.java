package org.deeplearning4j.module;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

/**
 * Inception is based on GoogleLeNet configuration of convolutional layers for optimization of
 * resources and learning. You can use this module to add Inception to your own custom models.
 *
 * The GoogleLeNet paper: https://arxiv.org/abs/1409.4842
 *
 * This module is based on the Inception GraphBuilderModule built for Torch and
 * a Scala implementation of GoogleLeNet.
 * https://github.com/Element-Research/dpnn/blob/master/Inception.lua
 * https://gist.github.com/antikantian/f77e91f924614348ea8f64731437930d
 *
 * @author Justin Long (crockpotveggies)
 */
public class Inception {

    public static String getModuleName() {
        return "inception";
    }

    public static String getModuleName(String layerName) { return getModuleName()+"-"+layerName; }


    public static ConvolutionLayer conv1x1(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] {1,1}, new int[] {1,1}, new int[] {0,0}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public static ConvolutionLayer c3x3reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    public static ConvolutionLayer c5x5reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    public static ConvolutionLayer conv3x3(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public static ConvolutionLayer conv5x5(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {2,2}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public static ConvolutionLayer conv7x7(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{7,7}, new int[]{2,2}, new int[]{3,3}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public static SubsamplingLayer avgPool7x7(int stride) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{7,7}, new int[]{1,1}).build();
    }

    public static SubsamplingLayer avgPoolNxN(int size, int stride) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{size,size}, new int[]{stride,stride}).build();
    }

    public static SubsamplingLayer maxPool3x3(int stride) {
        return new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{stride,stride}, new int[]{1,1}).build();
    }

    public static SubsamplingLayer maxPoolNxN(int size, int stride) {
        return new SubsamplingLayer.Builder(new int[]{size,size}, new int[]{stride,stride}, new int[]{1,1}).build();
    }

    public static DenseLayer fullyConnected(int in, int out, double dropOut) {
        return new DenseLayer.Builder().nIn(in).nOut(out).dropOut(dropOut).build();
    }

    public static ConvolutionLayer convNxN(int reduceSize, int outputSize, int kernelSize, int kernelStride) {
        return new ConvolutionLayer.Builder(new int[]{kernelSize,kernelSize}, new int[]{kernelStride,kernelStride}, new int[]{2,2}).nIn(reduceSize).nOut(outputSize).biasInit(0.2).build();
    }

    public static ConvolutionLayer convNxNreduce(int inputSize, int reduceSize, int reduceStride) {
        return new ConvolutionLayer.Builder(new int[]{1,1}, new int[]{reduceStride,reduceStride}).nIn(inputSize).nOut(reduceSize).biasInit(0.2).build();
    }

    public static BatchNormalization batchNorm(int in, int out) {
        return new BatchNormalization.Builder(false).nIn(in).nOut(in).activation("relu").build();
    }

    public static ComputationGraphConfiguration.GraphBuilder appendGraph(ComputationGraphConfiguration.GraphBuilder graph,
                                                                         String moduleLayerName, int inputSize, int[] kernelSize, int[] kernelStride, int[] outputSize, int[] reduceSize,
                                                                         SubsamplingLayer.PoolingType poolingType, boolean batchNorm, String inputLayer) {
        return appendGraph(graph,moduleLayerName,inputSize,kernelSize,kernelStride,outputSize,reduceSize,poolingType,3,1,batchNorm,inputLayer);
    }

    /**
     * Appends inception layer configurations a GraphBuilder object, based on the concept of
     * Inception via the GoogleLeNet paper: https://arxiv.org/abs/1409.4842
     *
     * @param graph An existing computation graph GraphBuilder object.
     * @param moduleLayerName The numerical order of inception (like 2, 2a, 3e, etc.)
     * @param inputSize
     * @param kernelSize
     * @param kernelStride
     * @param outputSize
     * @param reduceSize
     * @param poolingType
     * @param poolSize
     * @param poolStride
     * @param batchNorm
     * @param inputLayer
     * @return
     */
    public static ComputationGraphConfiguration.GraphBuilder appendGraph(ComputationGraphConfiguration.GraphBuilder graph,
                                                                         String moduleLayerName, int inputSize, int[] kernelSize, int[] kernelStride, int[] outputSize, int[] reduceSize,
                                                                         SubsamplingLayer.PoolingType poolingType, int poolSize, int poolStride, boolean batchNorm, String inputLayer) {
        // 1x1 reduce -> nxn conv
        for(int i=0; i<kernelSize.length; i++) {
            graph.addLayer(getModuleName(moduleLayerName)+"-cnn1-"+i, conv1x1(inputSize, reduceSize[i], 0.2), inputLayer);
            if(batchNorm) graph.addLayer(getModuleName(moduleLayerName)+"-batch1-"+i, batchNorm(reduceSize[i],reduceSize[i]), getModuleName(moduleLayerName)+"-cnn1-"+i);
            graph.addLayer(getModuleName(moduleLayerName)+"-reduce1-"+i, convNxN(reduceSize[i],outputSize[i],kernelSize[i],kernelStride[i]), batchNorm ? getModuleName(moduleLayerName)+"-batch1-"+i : getModuleName(moduleLayerName)+"-cnn1-"+i);
            if(batchNorm) graph.addLayer(getModuleName(moduleLayerName)+"-batch2-"+i, batchNorm(outputSize[i],outputSize[i]), getModuleName(moduleLayerName)+"-reduce1-"+i);
        }

        // pool -> 1x1 conv
        switch(poolingType) {
            case AVG:
                graph.addLayer(getModuleName(moduleLayerName)+"-pool1", avgPoolNxN(poolSize,poolStride), inputLayer);
                break;
            case MAX:
                graph.addLayer(getModuleName(moduleLayerName)+"-pool1", maxPoolNxN(poolSize,poolStride), inputLayer);
                break;
            default:
                throw new IllegalStateException("You must specify a valid pooling type of avg or max for Inception module.");
        }
        int i = kernelSize.length;
        try {
            graph.addLayer(getModuleName(moduleLayerName)+"-cnn2", convNxNreduce(inputSize,reduceSize[i],1), getModuleName(moduleLayerName)+"-pool1");
            if(batchNorm) graph.addLayer(getModuleName(moduleLayerName)+"-batch3", batchNorm(reduceSize[i],reduceSize[i]), getModuleName(moduleLayerName)+"-cnn2");
        }
        catch(IndexOutOfBoundsException e) {
            System.out.print(e.getStackTrace());
        }
        i++;

        // reduce
        try {
            graph.addLayer(getModuleName(moduleLayerName)+"-reduce2", convNxNreduce(inputSize,reduceSize[i],1), getModuleName(moduleLayerName)+"-pool1");
            if(batchNorm) graph.addLayer(getModuleName(moduleLayerName)+"-batch4", batchNorm(reduceSize[i],reduceSize[i]), getModuleName(moduleLayerName)+"-reduce2");
        }
        catch(IndexOutOfBoundsException e) {
            System.out.print(e);
        }

        // TODO: there's a better way to do this
        if(kernelSize.length==1 && reduceSize.length==3) {
            graph.addVertex(
                getModuleName(moduleLayerName),
                new MergeVertex(),
                batchNorm ? getModuleName(moduleLayerName) + "-batch2-0" : getModuleName(moduleLayerName) + "-reduce1-0",
                batchNorm ? getModuleName(moduleLayerName) + "-batch3" : getModuleName(moduleLayerName) + "-cnn2",
                batchNorm ? getModuleName(moduleLayerName) + "-batch4" : getModuleName(moduleLayerName) + "-reduce2"
            );
        }
        else if(kernelSize.length==2 && reduceSize.length==2) {
            graph.addVertex(
                getModuleName(moduleLayerName),
                new MergeVertex(),
                batchNorm ? getModuleName(moduleLayerName) + "-batch2-0" : getModuleName(moduleLayerName) + "-reduce1-0",
                batchNorm ? getModuleName(moduleLayerName) + "-batch2-1" : getModuleName(moduleLayerName) + "-reduce1-1"
            );
        }
        else if(kernelSize.length==2 && reduceSize.length==4) {
            graph.addVertex(
                getModuleName(moduleLayerName),
                new MergeVertex(),
                batchNorm ? getModuleName(moduleLayerName) + "-batch2-0" : getModuleName(moduleLayerName) + "-reduce1-0",
                batchNorm ? getModuleName(moduleLayerName) + "-batch2-1" : getModuleName(moduleLayerName) + "-reduce1-1",
                batchNorm ? getModuleName(moduleLayerName) + "-batch3" : getModuleName(moduleLayerName) + "-cnn2",
                batchNorm ? getModuleName(moduleLayerName) + "-batch4" : getModuleName(moduleLayerName) + "-reduce2"
            );
        }
        else throw new IllegalStateException("Only kernel of shape 1 or 2 and a reduce shape between 2 and 4 is supported.");

        return graph;
    }

}
