// network.js - Capas y modelo CNN desde cero
const Network = (function() {
    // ---------- Capa Convolucional ----------
    class Conv2D {
        constructor(numFilters, kernelSize, inputShape) {
            this.numFilters = numFilters;
            this.kernelSize = kernelSize;
            this.inputShape = inputShape; // [H, W, C]
            // Inicialización de pesos (He)
            const scale = Math.sqrt(2.0 / (kernelSize * kernelSize * inputShape[2]));
            this.weights = [];
            for (let f = 0; f < numFilters; f++) {
                const kernel = [];
                for (let ci = 0; ci < inputShape[2]; ci++) {
                    kernel.push([]);
                    for (let i = 0; i < kernelSize; i++) {
                        const row = [];
                        for (let j = 0; j < kernelSize; j++) {
                            row.push(gaussianRandom() * scale);
                        }
                        kernel[ci].push(row);
                    }
                }
                this.weights.push(kernel);
            }
            this.bias = new Array(numFilters).fill(0);
            // Espacio para almacenar entrada (para retropropagación)
            this.lastInput = null;
        }

        forward(input) {
            this.lastInput = input; // [H, W, C]
            const [H, W, C] = this.inputShape;
            const outH = H - this.kernelSize + 1;
            const outW = W - this.kernelSize + 1;
            const output = [];
            for (let f = 0; f < this.numFilters; f++) {
                const outChannel = [];
                for (let i = 0; i < outH; i++) {
                    outChannel.push(new Array(outW).fill(0));
                }
                for (let ci = 0; ci < C; ci++) {
                    const kernel = this.weights[f][ci];
                    const inChannel = input[ci];
                    for (let i = 0; i < outH; i++) {
                        for (let j = 0; j < outW; j++) {
                            let sum = 0;
                            for (let ki = 0; ki < this.kernelSize; ki++) {
                                for (let kj = 0; kj < this.kernelSize; kj++) {
                                    sum += inChannel[i+ki][j+kj] * kernel[ki][kj];
                                }
                            }
                            outChannel[i][j] += sum;
                        }
                    }
                }
                // agregar bias
                for (let i = 0; i < outH; i++) {
                    for (let j = 0; j < outW; j++) {
                        outChannel[i][j] += this.bias[f];
                    }
                }
                output.push(outChannel);
            }
            // Reorganizar a [outH][outW][f]
            const reshaped = [];
            for (let i = 0; i < outH; i++) {
                reshaped.push([]);
                for (let j = 0; j < outW; j++) {
                    reshaped[i].push(output.map(ch => ch[i][j]));
                }
            }
            return reshaped; // [H_out, W_out, F]
        }

        backward(dout) {
            // dout es [H_out, W_out, F]
            const [H, W, C] = this.inputShape;
            const outH = H - this.kernelSize + 1;
            const outW = W - this.kernelSize + 1;
            const F = this.numFilters;

            // Inicializar gradientes
            const dW = [];
            for (let f = 0; f < F; f++) {
                const dK = [];
                for (let ci = 0; ci < C; ci++) {
                    dK.push(Array(this.kernelSize).fill().map(() => Array(this.kernelSize).fill(0)));
                }
                dW.push(dK);
            }
            const db = new Array(F).fill(0);
            const dX = Array(C).fill().map(() => Array(H).fill().map(() => Array(W).fill(0)));

            const input = this.lastInput;

            for (let f = 0; f < F; f++) {
                for (let i = 0; i < outH; i++) {
                    for (let j = 0; j < outW; j++) {
                        const delta = dout[i][j][f];
                        db[f] += delta;
                        for (let ci = 0; ci < C; ci++) {
                            const inChannel = input[ci];
                            for (let ki = 0; ki < this.kernelSize; ki++) {
                                for (let kj = 0; kj < this.kernelSize; kj++) {
                                    dW[f][ci][ki][kj] += inChannel[i+ki][j+kj] * delta;
                                    dX[ci][i+ki][j+kj] += this.weights[f][ci][ki][kj] * delta;
                                }
                            }
                        }
                    }
                }
            }

            return { dX, dW, db };
        }
    }

    // ---------- ReLU ----------
    class ReLU {
        forward(input) {
            this.lastInput = input;
            return input.map(row => row.map(pixelRow => pixelRow.map(v => Math.max(0, v))));
        }
        backward(dout) {
            return dout.map((row, i) => row.map((pixRow, j) => pixRow.map((v, k) =>
                this.lastInput[i][j][k] > 0 ? v : 0
            )));
        }
    }

    // ---------- MaxPooling 2x2 ----------
    class MaxPool2D {
        forward(input) {
            this.lastInput = input;
            const [H, W, C] = [input.length, input[0].length, input[0][0].length];
            const outH = Math.floor(H / 2);
            const outW = Math.floor(W / 2);
            this.switches = []; // guardar índices del máximo para backward
            const output = [];
            for (let c = 0; c < C; c++) {
                const outCh = [];
                const switchCh = [];
                for (let i = 0; i < outH; i++) {
                    outCh.push([]);
                    switchCh.push([]);
                    for (let j = 0; j < outW; j++) {
                        let maxVal = -Infinity;
                        let maxPos = [0,0];
                        for (let di = 0; di < 2; di++) {
                            for (let dj = 0; dj < 2; dj++) {
                                const val = input[i*2+di][j*2+dj][c];
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxPos = [i*2+di, j*2+dj];
                                }
                            }
                        }
                        outCh[i].push(maxVal);
                        switchCh[i].push(maxPos);
                    }
                }
                output.push(outCh);
                this.switches.push(switchCh);
            }
            // Reorganizar a [outH][outW][C]
            const reshaped = [];
            for (let i = 0; i < outH; i++) {
                reshaped.push([]);
                for (let j = 0; j < outW; j++) {
                    reshaped[i].push(output.map(ch => ch[i][j]));
                }
            }
            return reshaped;
        }

        backward(dout) {
            const [H, W, C] = [this.lastInput.length, this.lastInput[0].length, this.lastInput[0][0].length];
            const outH = dout.length;
            const outW = dout[0].length;
            const dX = Array(H).fill().map(() => Array(W).fill().map(() => new Array(C).fill(0)));
            for (let i = 0; i < outH; i++) {
                for (let j = 0; j < outW; j++) {
                    for (let c = 0; c < C; c++) {
                        const [maxI, maxJ] = this.switches[c][i][j];
                        dX[maxI][maxJ][c] += dout[i][j][c];
                    }
                }
            }
            return dX;
        }
    }

    // ---------- Aplanado (Flatten) ----------
    class Flatten {
        forward(input) {
            this.origShape = [input.length, input[0].length, input[0][0].length];
            const flat = [];
            for (let i = 0; i < input.length; i++) {
                for (let j = 0; j < input[0].length; j++) {
                    for (let c = 0; c < input[0][0].length; c++) {
                        flat.push(input[i][j][c]);
                    }
                }
            }
            return flat;
        }
        backward(dout) {
            const [H, W, C] = this.origShape;
            const reshaped = Array(H).fill().map(() => Array(W).fill().map(() => new Array(C).fill(0)));
            let idx = 0;
            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    for (let c = 0; c < C; c++) {
                        reshaped[i][j][c] = dout[idx++];
                    }
                }
            }
            return reshaped;
        }
    }

    // ---------- Capa Densa ----------
    class Dense {
        constructor(inputSize, outputSize, activation = 'relu') {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.activation = activation;
            const scale = Math.sqrt(2.0 / inputSize);
            this.weights = Array(inputSize).fill().map(() =>
                Array(outputSize).fill().map(() => gaussianRandom() * scale)
            );
            this.bias = new Array(outputSize).fill(0);
        }

        forward(input) {
            this.lastInput = input;
            const out = new Array(this.outputSize).fill(0);
            for (let j = 0; j < this.outputSize; j++) {
                for (let i = 0; i < this.inputSize; i++) {
                    out[j] += input[i] * this.weights[i][j];
                }
                out[j] += this.bias[j];
            }
            this.z = out; // antes de activación
            if (this.activation === 'relu') {
                this.activated = out.map(v => Math.max(0, v));
                return this.activated;
            } else if (this.activation === 'softmax') {
                const maxVal = Math.max(...out);
                const expVals = out.map(v => Math.exp(v - maxVal));
                const sumExp = expVals.reduce((a,b)=>a+b,0);
                this.activated = expVals.map(v => v / sumExp);
                return this.activated;
            }
            return out;
        }

        backward(dout) {
            let dz;
            if (this.activation === 'relu') {
                dz = dout.map((d, i) => (this.z[i] > 0 ? d : 0));
            } else if (this.activation === 'softmax') {
                // Asumimos que la pérdida es cross-entropy y dout ya es (softmax - y)
                dz = dout.slice();
            } else {
                dz = dout.slice();
            }

            const dW = Array(this.inputSize).fill().map(() => new Array(this.outputSize).fill(0));
            const db = new Array(this.outputSize).fill(0);
            const dX = new Array(this.inputSize).fill(0);

            for (let j = 0; j < this.outputSize; j++) {
                db[j] += dz[j];
                for (let i = 0; i < this.inputSize; i++) {
                    dW[i][j] += this.lastInput[i] * dz[j];
                    dX[i] += this.weights[i][j] * dz[j];
                }
            }
            return { dX, dW, db };
        }
    }

    // ---------- Dropout ----------
    class Dropout {
        constructor(keepProb) {
            this.keepProb = keepProb;
            this.mask = null;
        }
        forward(input, training = true) {
            if (!training) return input;
            this.mask = input.map(() => (Math.random() < this.keepProb ? 1.0 / this.keepProb : 0));
            return input.map((v, i) => v * this.mask[i]);
        }
        backward(dout) {
            if (!this.mask) return dout;
            return dout.map((v, i) => v * this.mask[i]);
        }
    }

    // ---------- Modelo CNN ----------
    class CNNModel {
        constructor(config) {
            this.conv = new Conv2D(config.filters, 3, [8,8,1]);
            this.relu = new ReLU();
            this.pool = new MaxPool2D();
            this.flatten = new Flatten();
            const flatSize = 3 * 3 * config.filters;
            this.dense1 = new Dense(flatSize, 10, 'relu');
            this.dropout = new Dropout(config.dropoutKeep);
            this.dense2 = new Dense(10, 4, 'softmax');
        }

        forward(inputImg, training = true) {
            const inp = [inputImg]; // canal único
            const convOut = this.conv.forward(inp);
            const reluOut = this.relu.forward(convOut);
            const poolOut = this.pool.forward(reluOut);
            const flat = this.flatten.forward(poolOut);
            const dense1Out = this.dense1.forward(flat);
            const dropOut = this.dropout.forward(dense1Out, training);
            const out = this.dense2.forward(dropOut);
            return out;
        }

        backward(dout, lr) {
            // dout es el gradiente de softmax+crossentropy = softmax_output - y_true
            const gradDense2 = this.dense2.backward(dout);
            const gradDrop = this.dropout.backward(gradDense2.dX);
            const gradDense1 = this.dense1.backward(gradDrop);
            const gradFlat = this.flatten.backward(gradDense1.dX);
            const gradPool = this.pool.backward(gradFlat);
            const gradRelu = this.relu.backward(gradPool);
            const gradConv = this.conv.backward(gradRelu);
            return {
                conv: gradConv,
                dense1: { dW: gradDense1.dW, db: gradDense1.db },
                dense2: { dW: gradDense2.dW, db: gradDense2.db }
            };
        }
    }

    // Generador Gaussiano simple
    function gaussianRandom(mean=0, stdev=1) {
        let u = 1 - Math.random();
        let v = Math.random();
        let z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
        return z * stdev + mean;
    }

    return { Conv2D, ReLU, MaxPool2D, Flatten, Dense, Dropout, CNNModel, gaussianRandom };
})();