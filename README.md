# Image Morph

## Credits

Inspired by [this project](https://github.com/Spu7Nix/obamify).

## Demonstration

![Sample program output](/res/demo_anim.gif)

## Directory

| Type | Folder | Contents |
|---------|---------|-----------|
| Executables | `/bin` | Compiled transportation algorithm binary |
| Input data | `/data` | Raw inputs and generated I/O text/images |
| User assets | `/img` | User-imported images |
| Outputs | `/res` | Intermediate + final results |
| Source code | `/src` | Python scripts and `transport.cpp` |

## Usage (Local)

- Clone repository locally
- Navigate to `/src` directory
- If necessary, run `g++ -o ../bin/transport transport.cpp`
- Run `python3 main.py` to start the program

## Technical Details

The program can be roughly partitioned into three steps:
1. Reading and processing image data
2. Central transportation algorithm
3. Rendering animation

### Data Processing
The program loads `/data/target.png` (and `/data/weights.png` if specified) and reads them into a $N^2\times3$ flattened pixel array storing RGB data (any alpha channel information is lost).

The program then prompts the user for an image which is then cropped to match `target.png` aspect ratio. Upscaling is performed using the `NEAREST` resampling method, while downscaling uses the `LANCZOS` algorithm.

### Min-Cost Matching
The program seeks to match each original image pixel $(x,y)$ with a target pixel $(x',y')$. The cost of doing so is defined heuristically as
```math
c((x,y), (x',y')) = ((x-x')^2+(y-y')^2)^2 + w(x',y')\times ((r(x,y)-r(x',y'))^2 + (g(x,y)-g(x',y'))^2 + (b(x,y)-b(x',y'))^2)
```
where $w(x,y)$ denotes the weight of target pixel at $(x,y)$, $r(x,y)$ denotes the red channel of pixel at $(x,y)$, etc. In other words, the cost is the sum of the spatial Euclidean distance to the power of 4 with the squared Euclidean distance in RGB space.

The graph of all these possible matchable pairs produces a bipartite graph with $2N^2$ nodes. Common flow algorithms to solve this problem have a $O(V^3)$ running time in the number of vertices, which is too slow. Therefore, we take a heuristic approach.

The core of the algorithm is [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). We set a high initial "temperature". In essence, at every iteration, we pick two random nodes, and swap them if swapping them weakly decreases our cost. Else, we swap them with a probability corresponding to the current system temperature. Then, we decrease our temperature.

Over several iterations, we are likely to converge close to a global optima (or at least some local one).

### Animation
Each pixel knows its target location. The animation function upscales the image to provide subpixel accuracy before linearly interpolating each pixel to its target location across several frames.
