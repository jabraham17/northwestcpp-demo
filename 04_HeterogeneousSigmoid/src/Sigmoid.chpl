use GpuDiagnostics;
use Random;
use Math;
use BlockDist;

proc sigmoid(arr: [?dom] ?eltType, gpuPercent) {
  var result: arr.type;

  /* For each target locale
      compute a chunk of data and move computation to that locale
  */
  coforall cpuLocale in dom.targetLocales() do on cpuLocale {
    const localNodeDom = dom.localSubdomain();

    /* Split the data between the CPUs and GPUs, then launch both in parallel */
    const localN = localNodeDom.size;
    const localGpuN = (localN * gpuPercent):int;
    cobegin {
      sigmoidGpu(result, arr, localNodeDom[         ..<localGpuN], here.gpus);
      sigmoidCpu(result, arr, localNodeDom[localGpuN..          ]);
    }
  }
  return result;
}

/* `Xi = 1 / (1 + exp(-Xi))` */
inline proc sigmoidHelper(ref result, arr) {
  result = 1 / (1 + exp(-arr));
}

proc sigmoidCpu(ref result, arr, dom) {
  sigmoidHelper(result[dom], arr[dom]);
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
config const gpuPercent = 0.5;
proc main() {
  var Arr = blockDist.createArray({0..<n}, real);

  fillRandom(Arr, -6, 6);

  if verboseGpu then startVerboseGpu();
  var Result = sigmoid(Arr, gpuPercent);
  if verboseGpu then stopVerboseGpu();

  if print then writeln(Result);
  if printForGraph {
    for i in 0..<n {
      writeln(Arr[i], " ", Result[i]);
    }
  }
}
