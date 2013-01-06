int main() {
    write_pair(1, "threadIdx.y: ", threadIdx.y);
    write_pair(1, "threadIdx.x: ", threadIdx.x);
    write_pair(1, "threadDim.y: ", threadDim.y);
    write_pair(1, "threadDim.x: ", threadDim.x);
    write_pair(1, "blockIdx.y: ", blockIdx.y);
    write_pair(1, "blockIdx.x: ", blockIdx.x);
    write_pair(1, "blockDim.y: ", blockDim.y);
    write_pair(1, "blockDim.x: ", blockDim.x);
    return 0;
}
