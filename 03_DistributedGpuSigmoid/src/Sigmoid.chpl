use GpuDiagnostics;
use Random;
use Math;
use BlockDist;

proc sigmoid(arr: [?dom] ?eltType) {
  var result: arr.type;

  /* For each target locale
      compute a chunk of data and move computation to that locale
  */
  coforall cpuLocale in dom.targetLocales() do on cpuLocale {
    /* Determine the local indices */
    const localNodeDom = dom.localSubdomain();
    /* Same multi-GPU code as before - just wrapped in a function */
    sigmoidGpu(result, arr, localNodeDom, here.gpus);
  }
  return result;
}

/* `Xi = 1 / (1 + exp(-Xi))` */
inline proc sigmoidHelper(ref result, arr) {
  result = 1 / (1 + exp(-arr));
}

proc sigmoidGpu(ref result, arr, dom, tgtLoc) {
  import RangeChunk.chunks;
  coforall (loc, c) in zip(tgtLoc, chunks(dom.dim(0), tgtLoc.size)) do on loc {
    const localGpuDom = dom[c];
    var gpuArr = arr[localGpuDom];
    var gpuResult: [localGpuDom] arr.eltType;
    sigmoidHelper(gpuResult, gpuArr);
    result[localGpuDom] = gpuResult;
  }
}


config const n = 100;
config const print = false;
config const printForGraph = false;
config const verboseGpu = false;
proc main() {
  var Arr = blockDist.createArray({0..<n}, real);

  fillRandom(Arr, -6, 6);

  if verboseGpu then startVerboseGpu();
  var Result = sigmoid(Arr);
  if verboseGpu then stopVerboseGpu();

  if print then writeln(Result);
  if printForGraph {
    for i in 0..<n {
      writeln(Arr[i], " ", Result[i]);
    }
  }
}
