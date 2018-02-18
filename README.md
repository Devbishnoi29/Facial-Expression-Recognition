# Facial-Expression-Recognition using Tensorflow

This project uses tensorflow library to implement convolution neural network model.
This model is trained on facial-expression data set available on kaggle.
Here we facial expression model recognizes 6 basic emotions: happy, sad, surprise, fear, anger and neutral.
This data set contains about 28672 training images and 7168 testing images(public test + private test).
We are training our model on training images and validating our trained model on testing images.
In each epoch we are processing 35500 images of 48*48*3 dimentions.

Here we have cost, training accuracy, test accuracy with training epochs.

epoch :  1  cost:  15165665335.0   train acc:  7213 / 28672  test acc:  1684 / 7168
epoch :  2  cost:  7236739576.0   train acc:  8552 / 28672  test acc:  1964 / 7168
epoch :  3  cost:  5321358556.0   train acc:  8918 / 28672  test acc:  2018 / 7168
epoch :  4  cost:  4317758495.5   train acc:  8783 / 28672  test acc:  1928 / 7168
epoch :  5  cost:  3598166824.5   train acc:  8936 / 28672  test acc:  1951 / 7168
epoch :  6  cost:  3065639233.75   train acc:  9385 / 28672  test acc:  2019 / 7168
epoch :  7  cost:  2632670139.0   train acc:  9634 / 28672  test acc:  1980 / 7168
epoch :  8  cost:  2310305381.75   train acc:  10488 / 28672  test acc:  2112 / 7168
epoch :  9  cost:  2029115268.75   train acc:  11879 / 28672  test acc:  2291 / 7168
epoch :  10  cost:  1777792137.25   train acc:  12186 / 28672  test acc:  2317 / 7168
epoch :  11  cost:  1556510014.125   train acc:  12985 / 28672  test acc:  2349 / 7168
epoch :  12  cost:  1359082716.875   train acc:  13442 / 28672  test acc:  2417 / 7168
epoch :  13  cost:  1199721577.0   train acc:  13915 / 28672  test acc:  2477 / 7168
epoch :  14  cost:  1048370381.8125   train acc:  14365 / 28672  test acc:  2478 / 7168
epoch :  15  cost:  935349166.0625   train acc:  15088 / 28672  test acc:  2542 / 7168
epoch :  16  cost:  838720616.9375   train acc:  15392 / 28672  test acc:  2612 / 7168
epoch :  17  cost:  777195333.375   train acc:  14706 / 28672  test acc:  2460 / 7168
epoch :  18  cost:  711690754.625   train acc:  15445 / 28672  test acc:  2443 / 7168
epoch :  19  cost:  616924073.25   train acc:  16259 / 28672  test acc:  2459 / 7168
epoch :  20  cost:  544953402.125   train acc:  17858 / 28672  test acc:  2539 / 7168
epoch :  21  cost:  481028458.46875   train acc:  19061 / 28672  test acc:  2626 / 7168
epoch :  22  cost:  408414318.3125   train acc:  19550 / 28672  test acc:  2695 / 7168
epoch :  23  cost:  365802783.40625   train acc:  19084 / 28672  test acc:  2755 / 7168
epoch :  24  cost:  326591756.203125   train acc:  19601 / 28672  test acc:  2784 / 7168
epoch :  25  cost:  282989052.515625   train acc:  19556 / 28672  test acc:  2747 / 7168
epoch :  26  cost:  249887212.234375   train acc:  21360 / 28672  test acc:  2797 / 7168
epoch :  27  cost:  241522502.140625   train acc:  21476 / 28672  test acc:  2719 / 7168
epoch :  28  cost:  228091611.15625   train acc:  22225 / 28672  test acc:  2768 / 7168
epoch :  29  cost:  214492489.4140625   train acc:  21941 / 28672  test acc:  2812 / 7168
epoch :  30  cost:  185023736.53125   train acc:  22288 / 28672  test acc:  2830 / 7168
epoch :  31  cost:  164348103.546875   train acc:  21794 / 28672  test acc:  2777 / 7168
epoch :  32  cost:  130319632.453125   train acc:  21618 / 28672  test acc:  2669 / 7168
epoch :  33  cost:  102288892.41015625   train acc:  23273 / 28672  test acc:  2714 / 7168
epoch :  34  cost:  89067810.27539062   train acc:  23378 / 28672  test acc:  2749 / 7168
epoch :  35  cost:  79164715.76367188   train acc:  22855 / 28672  test acc:  2674 / 7168
epoch :  36  cost:  68351617.78808594   train acc:  24177 / 28672  test acc:  2839 / 7168
epoch :  37  cost:  60699066.79589844   train acc:  23750 / 28672  test acc:  2771 / 7168
epoch :  38  cost:  56546747.91894531   train acc:  24216 / 28672  test acc:  2841 / 7168
epoch :  39  cost:  49596428.150390625   train acc:  25132 / 28672  test acc:  2898 / 7168
epoch :  40  cost:  47576223.650390625   train acc:  24462 / 28672  test acc:  2866 / 7168
epoch :  41  cost:  45941410.72949219   train acc:  22141 / 28672  test acc:  2666 / 7168
epoch :  42  cost:  41866706.447753906   train acc:  21760 / 28672  test acc:  2738 / 7168
epoch :  43  cost:  38701638.596191406   train acc:  24701 / 28672  test acc:  2887 / 7168
epoch :  44  cost:  33630342.30859375   train acc:  23783 / 28672  test acc:  2828 / 7168
epoch :  45  cost:  31964901.98803711   train acc:  24122 / 28672  test acc:  2858 / 7168
epoch :  46  cost:  30847160.69592285   train acc:  25377 / 28672  test acc:  2896 / 7168
epoch :  47  cost:  26159882.752075195   train acc:  26115 / 28672  test acc:  2970 / 7168
epoch :  48  cost:  21165810.879882812   train acc:  26839 / 28672  test acc:  3005 / 7168
epoch :  49  cost:  20693604.372802734   train acc:  26976 / 28672  test acc:  3059 / 7168
epoch :  50  cost:  18629499.391845703   train acc:  27025 / 28672  test acc:  3044 / 7168
epoch :  51  cost:  16997165.002685547   train acc:  26353 / 28672  test acc:  3037 / 7168
epoch :  52  cost:  16464686.280395508   train acc:  25517 / 28672  test acc:  2926 / 7168
epoch :  53  cost:  15079002.959075928   train acc:  26078 / 28672  test acc:  2988 / 7168
epoch :  54  cost:  14237528.275878906   train acc:  27140 / 28672  test acc:  3042 / 7168
epoch :  55  cost:  15442032.483032227   train acc:  25877 / 28672  test acc:  2975 / 7168
epoch :  56  cost:  14791048.852111816   train acc:  26700 / 28672  test acc:  3055 / 7168
epoch :  57  cost:  15572031.926635742   train acc:  25936 / 28672  test acc:  2940 / 7168
epoch :  58  cost:  14081584.420856476   train acc:  27417 / 28672  test acc:  3016 / 7168
epoch :  59  cost:  13518555.328493861   train acc:  26540 / 28672  test acc:  3046 / 7168
epoch :  60  cost:  11646346.450946808   train acc:  26642 / 28672  test acc:  3104 / 7168
epoch :  61  cost:  12647372.4678154   train acc:  27188 / 28672  test acc:  3041 / 7168
epoch :  62  cost:  10647762.420471191   train acc:  26833 / 28672  test acc:  3130 / 7168
epoch :  63  cost:  10231502.643675804   train acc:  27531 / 28672  test acc:  3092 / 7168
epoch :  64  cost:  12382367.184242249   train acc:  26442 / 28672  test acc:  2967 / 7168
epoch :  65  cost:  12990593.242847443   train acc:  26477 / 28672  test acc:  3013 / 7168
epoch :  66  cost:  10382846.068695068   train acc:  26796 / 28672  test acc:  3092 / 7168
epoch :  67  cost:  10279110.340015411   train acc:  26945 / 28672  test acc:  2950 / 7168
epoch :  68  cost:  10724515.887840271   train acc:  26881 / 28672  test acc:  3127 / 7168
epoch :  69  cost:  10796070.252212582   train acc:  27219 / 28672  test acc:  3080 / 7168
epoch :  70  cost:  10626302.272785187   train acc:  27827 / 28672  test acc:  3118 / 7168
epoch :  71  cost:  10990408.931396484   train acc:  26839 / 28672  test acc:  3091 / 7168
epoch :  72  cost:  9814645.451957703   train acc:  27828 / 28672  test acc:  3130 / 7168
epoch :  73  cost:  10159602.016139984   train acc:  27660 / 28672  test acc:  3112 / 7168
epoch :  74  cost:  10128269.267658234   train acc:  27716 / 28672  test acc:  3146 / 7168
epoch :  75  cost:  10649866.49369812   train acc:  27553 / 28672  test acc:  3123 / 7168
epoch :  76  cost:  9167193.421764374   train acc:  27296 / 28672  test acc:  3089 / 7168
epoch :  77  cost:  10267554.826644897   train acc:  27965 / 28672  test acc:  3076 / 7168
epoch :  78  cost:  9133681.758714676   train acc:  27891 / 28672  test acc:  3091 / 7168
epoch :  79  cost:  9032810.867706299   train acc:  27578 / 28672  test acc:  2988 / 7168
epoch :  80  cost:  10335012.695072174   train acc:  27576 / 28672  test acc:  3048 / 7168
epoch :  81  cost:  8955881.10490799   train acc:  27904 / 28672  test acc:  3121 / 7168
epoch :  82  cost:  8294377.22284317   train acc:  27454 / 28672  test acc:  3141 / 7168
epoch :  83  cost:  8969452.417964935   train acc:  27593 / 28672  test acc:  3079 / 7168
epoch :  84  cost:  8489518.256103516   train acc:  27721 / 28672  test acc:  3088 / 7168
epoch :  85  cost:  8002541.050621033   train acc:  27980 / 28672  test acc:  3148 / 7168
epoch :  86  cost:  8600121.629581451   train acc:  27421 / 28672  test acc:  2968 / 7168
epoch :  87  cost:  8891165.505649919   train acc:  27714 / 28672  test acc:  3110 / 7168
epoch :  88  cost:  9183912.639331818   train acc:  27823 / 28672  test acc:  3150 / 7168
epoch :  89  cost:  7991417.71913147   train acc:  27624 / 28672  test acc:  3136 / 7168
epoch :  90  cost:  8432556.904781342   train acc:  27531 / 28672  test acc:  3123 / 7168
epoch :  91  cost:  8330354.230007172   train acc:  27831 / 28672  test acc:  3130 / 7168
epoch :  92  cost:  7032910.414138794   train acc:  27447 / 28672  test acc:  3087 / 7168
epoch :  93  cost:  8984650.18371582   train acc:  26984 / 28672  test acc:  3078 / 7168
epoch :  94  cost:  7760473.040641785   train acc:  27513 / 28672  test acc:  3099 / 7168
epoch :  95  cost:  8099285.912059784   train acc:  26080 / 28672  test acc:  3063 / 7168
epoch :  96  cost:  7797306.575397491   train acc:  27369 / 28672  test acc:  3092 / 7168
epoch :  97  cost:  7995393.507463455   train acc:  27049 / 28672  test acc:  3085 / 7168
epoch :  98  cost:  7778180.11340332   train acc:  27994 / 28672  test acc:  3125 / 7168
epoch :  99  cost:  6883788.428237915   train acc:  27879 / 28672  test acc:  3086 / 7168
epoch :  100  cost:  7296276.241004944   train acc:  27745 / 28672  test acc:  3136 / 7168
