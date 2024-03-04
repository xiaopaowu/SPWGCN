<!--
 * @Author: Wuyifan
 * @Date: 2023-10-14 12:45:22
 * @LastEditors: Wuyifan
 * @LastEditTime: 2024-03-04 23:51:13
-->

## Reproducing Results

1. Run `python create_my_data.py YZOffice`
2. Run `python remove_words.py YZOffice 0.8`
3. Run `python build_graph.py YZOffice 0.8`
4. Run `python train.py YZOffice 0.8`
5. Change `YZOffice 0.8` in above 3 command lines when producing results for other datasets.