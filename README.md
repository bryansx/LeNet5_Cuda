# LeNet5_Cuda
TP Hardware for Signal Processing

## Partie 1
1.1- Creation de fonctions basiques de manipulations de matrices en C et en Cuda

1.2- Comparaison de temps de calcul pour l'addition parallélisée et non parallélisée avec "addMatC" et "addMatCuda" qui prennent en argument la taille des matrices.

1.2.1- Pour des matrices de taille 10000*1000 on obtient une accélération de (45604/22), donc d'environ 2000. Ce qui reste dans l'ordre de grandeur de l'accélération th. 

1.3- Comparaison de temps de calcul pour la multiplication parallélisée et non parallélisée avec "mulMatC" et "mulMatCuda" qui prennent en argument la taille des matrices.

1.3.1 On obtient une accélération bien plus importante pour les multiplications de matrices: 9512 ms pour le CPU contre 0.025 pour le GPU, soit une accélération de 38000 environ.

## Partie 2
2.1- Convolution 2D avec 6 noyaux de conv de taille 5x5. La taille résultantes est donc de 6x28x28.

2.1.1- Nous allons dans un premier temps écrire et tester la fonction de convolution afin de s'assurer quelle se comporte comme prévu. Pour se faire nous utiliser une image de la database MNIST. Puis nous allons faire une convolution avec le filtre de Sobel afin de vérifier que nous avons une bonne détection des contours. (run ./test_conv2d_MNIST et ./test_conv2d_squarre).
On peut bien voir sur les resultats de la conv du carré que les lignes verticales en disparues et les contours horizontaux ont été détectés.

=> La fonction conv2d semble fonctionner comme il faut.

2.2 Travail sur le fichier "Partie2/CNN.cu": 





