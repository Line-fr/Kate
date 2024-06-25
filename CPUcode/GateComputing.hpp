#ifndef CPUGATECOMPUTINGDONE
#define CPUGATECOMPUTINGDONE

namespace Kate{

void computeGate(Gate gate, int nqbits, Complex* qbitsstateshared, int* bit_to_groupbitnumber, std::vector<int> ordered_qbits, std::vector<Complex> cache){
    cache.resize((1llu << gate.qbits.size()));
    switch(gate.identifier){
        case TOFFOLI: {
            size_t to_cover = (1llu << (nqbits - 3));
            //we don't even need to put the gate in memory since it s not dense, let's get our indexes
            int lq0 = bit_to_groupbitnumber[gate.qbits[0]];
            int lq1 = bit_to_groupbitnumber[gate.qbits[1]];
            int lq2 = bit_to_groupbitnumber[gate.qbits[2]];
            
            size_t mask0, mask1, mask2, mask3;
            mask0 = (1llu << (bit_to_groupbitnumber[ordered_qbits[0]])) - 1;
            mask1 = (1llu << (bit_to_groupbitnumber[ordered_qbits[1]] - 1)) - 1 - mask0;
            mask2 = (1llu << (bit_to_groupbitnumber[ordered_qbits[2]] - 2)) - 1 - mask0 - mask1;
            mask3 = (1llu << (nqbits-3)) - 1 - mask0 - mask1 - mask2;
            for (size_t line = 0; line < to_cover; line++){
                size_t index110 = (1llu << lq0) + (1llu << lq1) + (line&mask0) + ((line&mask1) << (1)) + ((line&mask2) << (2)) + ((line&mask3) << (3)); //XXXXX-lq1(1)-XXXXX-lq0(0)-XXXXX
                size_t index111 = index110 + (1llu << lq2);
                auto temp = qbitsstateshared[index110];
                qbitsstateshared[index110] = qbitsstateshared[index111];
                qbitsstateshared[index111] = temp;
            }
            break;
        }
        case CRk: {
            size_t to_cover = (1llu << (nqbits - 2));
            //we don't even need to put the gate in memory since it s not dense, let's get our indexes
            int lq0 = bit_to_groupbitnumber[gate.qbits[0]];
            int lq1 = bit_to_groupbitnumber[gate.qbits[1]];
            size_t mask0, mask1, mask2;
            mask0 = (1llu << (bit_to_groupbitnumber[ordered_qbits[0]])) - 1;
            mask1 = (1llu << (bit_to_groupbitnumber[ordered_qbits[1]] - 1)) - 1 - mask0;
            mask2 = (1llu << (nqbits-2)) - 1 - mask0 - mask1;
            for (size_t line = 0; line < to_cover; line++){
                size_t index10 = (1llu << lq1) + (line&mask0) + ((line&mask1) << (1)) + ((line&mask2) << (2)); //XXXXX-lq1(1)-XXXXX-lq0(0)-XXXXX
                size_t index11 = index10 + (1llu << lq0);
                double temp = ((double)2*PI)/(1llu << (gate.optarg));
                qbitsstateshared[index11] *= Complex(cos(temp), sin(temp));
            }
            break;
        }
        case CNOT: {//CNOT
            //CNOT being a small gate, it is more interesting to make parallel the index of qbitstates
            size_t to_cover = (1llu << (nqbits - 2));
            //we don't even need to put the gate in memory since it s not dense, let's get our indexes
            int lq0 = bit_to_groupbitnumber[gate.qbits[0]];
            int lq1 = bit_to_groupbitnumber[gate.qbits[1]];
            size_t mask0, mask1, mask2;
            mask0 = (1llu << (bit_to_groupbitnumber[ordered_qbits[0]])) - 1;
            mask1 = (1llu << (bit_to_groupbitnumber[ordered_qbits[1]] - 1)) - 1 - mask0;
            mask2 = (1llu << (nqbits-2)) - 1 - mask0 - mask1;
            for (size_t line = 0; line < to_cover; line++){
                size_t index10 = (1llu << lq1) + (line&mask0) + ((line&mask1) << (1)) + ((line&mask2) << (2)); //XXXXX-lq1(1)-XXXXX-lq0(0)-XXXXX
                size_t index11 = index10 + (1llu << lq0);
                auto temp = qbitsstateshared[index10];
                qbitsstateshared[index10] = qbitsstateshared[index11];
                qbitsstateshared[index11] = temp;
            }
            break;
        }
        case Hadamard: { //H
            //H being a small gate, it is more interesting to make parallel the index of qbitstates
            size_t to_cover = (1llu << (nqbits - 1));
            //we don't even need to put the gate in memory since it s not dense, let's get our indexes
            int lq0 = bit_to_groupbitnumber[gate.qbits[0]];
            size_t mask0, mask1;
            mask0 = (1llu << (lq0)) - 1;
            mask1 = (1llu << (nqbits - 1)) - 1 - mask0;
            for (size_t line = 0; line < to_cover; line++){
                size_t index0 = (line&mask0) + ((line&mask1) << (1)); //XXXXX-lq0(0)-XXXXX
                size_t index1 = index0 + (1llu << lq0);
                //printf("index 0 : %llu , index 1 : %llu for lq0 : %i\n values before execution : %f, %f\n", index0, index1, lq0, qbitsstateshared[index0].a, qbitsstateshared[index1].a);
                qbitsstateshared[index0] = (qbitsstateshared[index0]+qbitsstateshared[index1])*SQRT2INV;
                qbitsstateshared[index1] = qbitsstateshared[index0] - qbitsstateshared[index1]*SQRT2;
                //printf("index 0 : %llu , index 1 : %llu for lq0 : %i\n values after execution : %f, %f\n", index0, index1, lq0, qbitsstateshared[index0].a, qbitsstateshared[index1].a);
            }
            break;
        }
        case 0: { //Dense
            //DENSE has 2 types of parallelization: matrix product and lines. ideally, we would use both of them to account for all matrix size cases.
            //for that purpose, let's parrallelize lines as much as possible, and allocate if there are some for the matrix product
            int gateqbits = gate.qbits.size();
            size_t to_cover = (1llu << (nqbits - gateqbits));
            //let's see if we can put this matrix in shared memory!
            Complex* matrixdata = gate.densecontent.data;
            //now it's time to build masks
            size_t masks[64]; //I want them in registers so better invoke them in a fixed size array. will work for as much as 63 qbits.
            size_t cumulative = 0;
            for (int i = 0; i < gateqbits; i++){
                masks[i] = (1llu << (bit_to_groupbitnumber[ordered_qbits[i]] - i)) - 1 - cumulative;
                cumulative += masks[i];
            }
            masks[gateqbits] = (1llu << (nqbits - gateqbits)) - 1 - cumulative;

            for (size_t line = 0; line < to_cover; line++){
                size_t baseind = 0; //will be XXXXX-0-XXXX-0-XXXXXX-0-...XXXX;
                for (int i = 0; i <= gateqbits; i++){
                    baseind += ((line&masks[i]) << i);
                }
                for (size_t matline = 0; matline < (1llu << gateqbits); matline++){
                    size_t tempind;
                    size_t lineind = baseind;
                    for (int i = 0; i < gateqbits; i++){
                        lineind += ((matline >> i)%2) << bit_to_groupbitnumber[gate.qbits[i]];
                    }
                    Complex sum = 0;
                    for (size_t matcol = 0; matcol < (1llu << gateqbits); matcol++){

                        tempind = baseind;
                        for (int i = 0; i < gateqbits; i++){
                            tempind += ((matcol >> i)%2) << bit_to_groupbitnumber[gate.qbits[i]];
                        }
                        sum += matrixdata[(matline << gateqbits) + matcol]*qbitsstateshared[tempind];
                    }
                    cache[matline] = sum;
                }
                for (size_t matline = 0; matline < (1llu << gateqbits); matline++){
                    size_t lineind = baseind;
                    for (int i = 0; i < gateqbits; i++){
                        lineind += ((matline >> i)%2) << bit_to_groupbitnumber[gate.qbits[i]];
                    }
                    qbitsstateshared[lineind] = cache[matline];
                }
            }
        }
    }
}

}

#endif