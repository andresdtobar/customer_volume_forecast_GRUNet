??,
? ?
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??*
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
: *
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
: *
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
: @*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
?
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	?H*
dtype0
?
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*.
shared_namegru/gru_cell/recurrent_kernel
?
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

:H*
dtype0
z
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*"
shared_namegru/gru_cell/bias
s
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:H*
dtype0
?
time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nametime_distributed/kernel
?
+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel*
_output_shapes

: *
dtype0
?
time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametime_distributed/bias
{
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes
: *
dtype0
?
time_distributed_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nametime_distributed_1/kernel
?
-time_distributed_1/kernel/Read/ReadVariableOpReadVariableOptime_distributed_1/kernel*
_output_shapes

: *
dtype0
?
time_distributed_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed_1/bias

+time_distributed_1/bias/Read/ReadVariableOpReadVariableOptime_distributed_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Nadam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv1d/kernel/m
?
)Nadam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d/kernel/m*"
_output_shapes
: *
dtype0
~
Nadam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/conv1d/bias/m
w
'Nadam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d/bias/m*
_output_shapes
: *
dtype0
?
Nadam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv1d_1/kernel/m
?
+Nadam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_1/kernel/m*"
_output_shapes
: @*
dtype0
?
Nadam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv1d_1/bias/m
{
)Nadam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_1/bias/m*
_output_shapes
:@*
dtype0
?
Nadam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*,
shared_nameNadam/gru/gru_cell/kernel/m
?
/Nadam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/kernel/m*
_output_shapes
:	?H*
dtype0
?
%Nadam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*6
shared_name'%Nadam/gru/gru_cell/recurrent_kernel/m
?
9Nadam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp%Nadam/gru/gru_cell/recurrent_kernel/m*
_output_shapes

:H*
dtype0
?
Nadam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H**
shared_nameNadam/gru/gru_cell/bias/m
?
-Nadam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/bias/m*
_output_shapes
:H*
dtype0
?
Nadam/time_distributed/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Nadam/time_distributed/kernel/m
?
3Nadam/time_distributed/kernel/m/Read/ReadVariableOpReadVariableOpNadam/time_distributed/kernel/m*
_output_shapes

: *
dtype0
?
Nadam/time_distributed/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameNadam/time_distributed/bias/m
?
1Nadam/time_distributed/bias/m/Read/ReadVariableOpReadVariableOpNadam/time_distributed/bias/m*
_output_shapes
: *
dtype0
?
!Nadam/time_distributed_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!Nadam/time_distributed_1/kernel/m
?
5Nadam/time_distributed_1/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/time_distributed_1/kernel/m*
_output_shapes

: *
dtype0
?
Nadam/time_distributed_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/time_distributed_1/bias/m
?
3Nadam/time_distributed_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/time_distributed_1/bias/m*
_output_shapes
:*
dtype0
?
Nadam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv1d/kernel/v
?
)Nadam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d/kernel/v*"
_output_shapes
: *
dtype0
~
Nadam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/conv1d/bias/v
w
'Nadam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d/bias/v*
_output_shapes
: *
dtype0
?
Nadam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv1d_1/kernel/v
?
+Nadam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_1/kernel/v*"
_output_shapes
: @*
dtype0
?
Nadam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv1d_1/bias/v
{
)Nadam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_1/bias/v*
_output_shapes
:@*
dtype0
?
Nadam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*,
shared_nameNadam/gru/gru_cell/kernel/v
?
/Nadam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/kernel/v*
_output_shapes
:	?H*
dtype0
?
%Nadam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*6
shared_name'%Nadam/gru/gru_cell/recurrent_kernel/v
?
9Nadam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp%Nadam/gru/gru_cell/recurrent_kernel/v*
_output_shapes

:H*
dtype0
?
Nadam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H**
shared_nameNadam/gru/gru_cell/bias/v
?
-Nadam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/bias/v*
_output_shapes
:H*
dtype0
?
Nadam/time_distributed/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Nadam/time_distributed/kernel/v
?
3Nadam/time_distributed/kernel/v/Read/ReadVariableOpReadVariableOpNadam/time_distributed/kernel/v*
_output_shapes

: *
dtype0
?
Nadam/time_distributed/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameNadam/time_distributed/bias/v
?
1Nadam/time_distributed/bias/v/Read/ReadVariableOpReadVariableOpNadam/time_distributed/bias/v*
_output_shapes
: *
dtype0
?
!Nadam/time_distributed_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!Nadam/time_distributed_1/kernel/v
?
5Nadam/time_distributed_1/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/time_distributed_1/kernel/v*
_output_shapes

: *
dtype0
?
Nadam/time_distributed_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/time_distributed_1/bias/v
?
3Nadam/time_distributed_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/time_distributed_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?K
value?JB?J B?J
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
l
(cell
)
state_spec
*regularization_losses
+	variables
,trainable_variables
-	keras_api
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
]
	2layer
3regularization_losses
4	variables
5trainable_variables
6	keras_api
]
	7layer
8regularization_losses
9	variables
:trainable_variables
;	keras_api
?
<iter

=beta_1

>beta_2
	?decay
@learning_rate
Amomentum_cachem?m?m?m?Bm?Cm?Dm?Em?Fm?Gm?Hm?v?v?v?v?Bv?Cv?Dv?Ev?Fv?Gv?Hv?
 
N
0
1
2
3
B4
C5
D6
E7
F8
G9
H10
N
0
1
2
3
B4
C5
D6
E7
F8
G9
H10
?
regularization_losses
Ilayer_metrics
	variables
trainable_variables
Jnon_trainable_variables
Klayer_regularization_losses
Lmetrics

Mlayers
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
Nlayer_metrics
	variables
trainable_variables
Onon_trainable_variables
Player_regularization_losses
Qmetrics

Rlayers
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
Slayer_metrics
	variables
trainable_variables
Tnon_trainable_variables
Ulayer_regularization_losses
Vmetrics

Wlayers
 
 
 
?
regularization_losses
Xlayer_metrics
	variables
trainable_variables
Ynon_trainable_variables
Zlayer_regularization_losses
[metrics

\layers
 
 
 
?
 regularization_losses
]layer_metrics
!	variables
"trainable_variables
^non_trainable_variables
_layer_regularization_losses
`metrics

alayers
 
 
 
?
$regularization_losses
blayer_metrics
%	variables
&trainable_variables
cnon_trainable_variables
dlayer_regularization_losses
emetrics

flayers
~

Bkernel
Crecurrent_kernel
Dbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
 
 

B0
C1
D2

B0
C1
D2
?
*regularization_losses
klayer_metrics
+	variables
,trainable_variables
lnon_trainable_variables
mlayer_regularization_losses

nstates
ometrics

players
 
 
 
?
.regularization_losses
qlayer_metrics
/	variables
0trainable_variables
rnon_trainable_variables
slayer_regularization_losses
tmetrics

ulayers
h

Ekernel
Fbias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
 

E0
F1

E0
F1
?
3regularization_losses
zlayer_metrics
4	variables
5trainable_variables
{non_trainable_variables
|layer_regularization_losses
}metrics

~layers
k

Gkernel
Hbias
regularization_losses
?	variables
?trainable_variables
?	keras_api
 

G0
H1

G0
H1
?
8regularization_losses
?layer_metrics
9	variables
:trainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEgru/gru_cell/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEgru/gru_cell/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEtime_distributed/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtime_distributed_1/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtime_distributed_1/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

B0
C1
D2

B0
C1
D2
?
gregularization_losses
?layer_metrics
h	variables
itrainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
 
 
 
 
 

(0
 
 
 
 
 
 

E0
F1

E0
F1
?
vregularization_losses
?layer_metrics
w	variables
xtrainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
 
 
 
 

20
 

G0
H1

G0
H1
?
regularization_losses
?layer_metrics
?	variables
?trainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
 
 
 
 

70
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}{
VARIABLE_VALUENadam/conv1d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/conv1d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUENadam/gru/gru_cell/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Nadam/gru/gru_cell/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUENadam/gru/gru_cell/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUENadam/time_distributed/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUENadam/time_distributed/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Nadam/time_distributed_1/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/time_distributed_1/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/conv1d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/conv1d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUENadam/gru/gru_cell/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Nadam/gru/gru_cell/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUENadam/gru/gru_cell/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUENadam/time_distributed/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUENadam/time_distributed/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Nadam/time_distributed_1/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/time_distributed_1/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv1d_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_inputconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasgru/gru_cell/kernelgru/gru_cell/biasgru/gru_cell/recurrent_kerneltime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_32232
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_1/kernel/Read/ReadVariableOp+time_distributed_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Nadam/conv1d/kernel/m/Read/ReadVariableOp'Nadam/conv1d/bias/m/Read/ReadVariableOp+Nadam/conv1d_1/kernel/m/Read/ReadVariableOp)Nadam/conv1d_1/bias/m/Read/ReadVariableOp/Nadam/gru/gru_cell/kernel/m/Read/ReadVariableOp9Nadam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp-Nadam/gru/gru_cell/bias/m/Read/ReadVariableOp3Nadam/time_distributed/kernel/m/Read/ReadVariableOp1Nadam/time_distributed/bias/m/Read/ReadVariableOp5Nadam/time_distributed_1/kernel/m/Read/ReadVariableOp3Nadam/time_distributed_1/bias/m/Read/ReadVariableOp)Nadam/conv1d/kernel/v/Read/ReadVariableOp'Nadam/conv1d/bias/v/Read/ReadVariableOp+Nadam/conv1d_1/kernel/v/Read/ReadVariableOp)Nadam/conv1d_1/bias/v/Read/ReadVariableOp/Nadam/gru/gru_cell/kernel/v/Read/ReadVariableOp9Nadam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp-Nadam/gru/gru_cell/bias/v/Read/ReadVariableOp3Nadam/time_distributed/kernel/v/Read/ReadVariableOp1Nadam/time_distributed/bias/v/Read/ReadVariableOp5Nadam/time_distributed_1/kernel/v/Read/ReadVariableOp3Nadam/time_distributed_1/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_34739
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachegru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biastotalcounttotal_1count_1Nadam/conv1d/kernel/mNadam/conv1d/bias/mNadam/conv1d_1/kernel/mNadam/conv1d_1/bias/mNadam/gru/gru_cell/kernel/m%Nadam/gru/gru_cell/recurrent_kernel/mNadam/gru/gru_cell/bias/mNadam/time_distributed/kernel/mNadam/time_distributed/bias/m!Nadam/time_distributed_1/kernel/mNadam/time_distributed_1/bias/mNadam/conv1d/kernel/vNadam/conv1d/bias/vNadam/conv1d_1/kernel/vNadam/conv1d_1/bias/vNadam/gru/gru_cell/kernel/v%Nadam/gru/gru_cell/recurrent_kernel/vNadam/gru/gru_cell/bias/vNadam/time_distributed/kernel/vNadam/time_distributed/bias/v!Nadam/time_distributed_1/kernel/vNadam/time_distributed_1/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_34878??)
?
I
-__inference_max_pooling1d_layer_call_fn_32994

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_301832
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?a
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_30351

inputs

states*
readvariableop_resource:	?H'
readvariableop_3_resource:H+
readvariableop_6_resource:H
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddz
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1z
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6r
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3~
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7r
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Muld
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1f
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1d
mul_2Mulclip_by_value_1:z:0states*
T0*'
_output_shapes
:?????????2
mul_2~
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:?????????2
Relub
mul_3Mulclip_by_value:z:0states*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subd
mul_4Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5d
IdentityIdentity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:??????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
??
?
>__inference_gru_layer_call_and_return_conditional_losses_33804

inputs3
 gru_cell_readvariableop_resource:	?H0
"gru_cell_readvariableop_3_resource:H4
"gru_cell_readvariableop_6_resource:H
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?gru_cell/ReadVariableOp_7?gru_cell/ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_2ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_2?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_1?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_2?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_6:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulzeros:output:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_7ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_7?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_7:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulzeros:output:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/adde
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Consti
gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_1?
gru_cell/MulMulgru_cell/add:z:0gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul?
gru_cell/Add_1AddV2gru_cell/Mul:z:0gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_1?
 gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_cell/clip_by_value/Minimum/y?
gru_cell/clip_by_value/MinimumMinimumgru_cell/Add_1:z:0)gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2 
gru_cell/clip_by_value/Minimumy
gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value/y?
gru_cell/clip_by_valueMaximum"gru_cell/clip_by_value/Minimum:z:0!gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value?
gru_cell/add_2AddV2gru_cell/BiasAdd_1:output:0gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_2i
gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Const_2i
gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_3?
gru_cell/Mul_1Mulgru_cell/add_2:z:0gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul_1?
gru_cell/Add_3AddV2gru_cell/Mul_1:z:0gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_3?
"gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru_cell/clip_by_value_1/Minimum/y?
 gru_cell/clip_by_value_1/MinimumMinimumgru_cell/Add_3:z:0+gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_cell/clip_by_value_1/Minimum}
gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value_1/y?
gru_cell/clip_by_value_1Maximum$gru_cell/clip_by_value_1/Minimum:z:0#gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value_1?
gru_cell/mul_2Mulgru_cell/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_8ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_8?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlice!gru_cell/ReadVariableOp_8:value:0'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_8?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_5?
gru_cell/add_4AddV2gru_cell/BiasAdd_2:output:0gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_4l
gru_cell/ReluRelugru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/Relu?
gru_cell/mul_3Mulgru_cell/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_3e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/sub?
gru_cell/mul_4Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_4?
gru_cell/add_5AddV2gru_cell/mul_3:z:0gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_3_resource"gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_33666*
condR
while_cond_33665*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1n
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^gru_cell/ReadVariableOp_7^gru_cell/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_626
gru_cell/ReadVariableOp_7gru_cell/ReadVariableOp_726
gru_cell/ReadVariableOp_8gru_cell/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
gru_while_cond_32383$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_32383___redundant_placeholder0;
7gru_while_gru_while_cond_32383___redundant_placeholder1;
7gru_while_gru_while_cond_32383___redundant_placeholder2;
7gru_while_gru_while_cond_32383___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
sequential_gru_while_body_30008:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_29
5sequential_gru_while_sequential_gru_strided_slice_1_0u
qsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0J
7sequential_gru_while_gru_cell_readvariableop_resource_0:	?HG
9sequential_gru_while_gru_cell_readvariableop_3_resource_0:HK
9sequential_gru_while_gru_cell_readvariableop_6_resource_0:H!
sequential_gru_while_identity#
sequential_gru_while_identity_1#
sequential_gru_while_identity_2#
sequential_gru_while_identity_3#
sequential_gru_while_identity_47
3sequential_gru_while_sequential_gru_strided_slice_1s
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorH
5sequential_gru_while_gru_cell_readvariableop_resource:	?HE
7sequential_gru_while_gru_cell_readvariableop_3_resource:HI
7sequential_gru_while_gru_cell_readvariableop_6_resource:H??,sequential/gru/while/gru_cell/ReadVariableOp?.sequential/gru/while/gru_cell/ReadVariableOp_1?.sequential/gru/while/gru_cell/ReadVariableOp_2?.sequential/gru/while/gru_cell/ReadVariableOp_3?.sequential/gru/while/gru_cell/ReadVariableOp_4?.sequential/gru/while/gru_cell/ReadVariableOp_5?.sequential/gru/while/gru_cell/ReadVariableOp_6?.sequential/gru/while/gru_cell/ReadVariableOp_7?.sequential/gru/while/gru_cell/ReadVariableOp_8?
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2H
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8sequential/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0 sequential_gru_while_placeholderOsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02:
8sequential/gru/while/TensorArrayV2Read/TensorListGetItem?
,sequential/gru/while/gru_cell/ReadVariableOpReadVariableOp7sequential_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02.
,sequential/gru/while/gru_cell/ReadVariableOp?
1sequential/gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1sequential/gru/while/gru_cell/strided_slice/stack?
3sequential/gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       25
3sequential/gru/while/gru_cell/strided_slice/stack_1?
3sequential/gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3sequential/gru/while/gru_cell/strided_slice/stack_2?
+sequential/gru/while/gru_cell/strided_sliceStridedSlice4sequential/gru/while/gru_cell/ReadVariableOp:value:0:sequential/gru/while/gru_cell/strided_slice/stack:output:0<sequential/gru/while/gru_cell/strided_slice/stack_1:output:0<sequential/gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2-
+sequential/gru/while/gru_cell/strided_slice?
$sequential/gru/while/gru_cell/MatMulMatMul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2&
$sequential/gru/while/gru_cell/MatMul?
.sequential/gru/while/gru_cell/ReadVariableOp_1ReadVariableOp7sequential_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_1?
3sequential/gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       25
3sequential/gru/while/gru_cell/strided_slice_1/stack?
5sequential/gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   27
5sequential/gru/while/gru_cell/strided_slice_1/stack_1?
5sequential/gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_1/stack_2?
-sequential/gru/while/gru_cell/strided_slice_1StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_1:value:0<sequential/gru/while/gru_cell/strided_slice_1/stack:output:0>sequential/gru/while/gru_cell/strided_slice_1/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_1?
&sequential/gru/while/gru_cell/MatMul_1MatMul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:06sequential/gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/gru/while/gru_cell/MatMul_1?
.sequential/gru/while/gru_cell/ReadVariableOp_2ReadVariableOp7sequential_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_2?
3sequential/gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   25
3sequential/gru/while/gru_cell/strided_slice_2/stack?
5sequential/gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5sequential/gru/while/gru_cell/strided_slice_2/stack_1?
5sequential/gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_2/stack_2?
-sequential/gru/while/gru_cell/strided_slice_2StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_2:value:0<sequential/gru/while/gru_cell/strided_slice_2/stack:output:0>sequential/gru/while/gru_cell/strided_slice_2/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_2?
&sequential/gru/while/gru_cell/MatMul_2MatMul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:06sequential/gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/gru/while/gru_cell/MatMul_2?
.sequential/gru/while/gru_cell/ReadVariableOp_3ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_3?
3sequential/gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/gru/while/gru_cell/strided_slice_3/stack?
5sequential/gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_3/stack_1?
5sequential/gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_3/stack_2?
-sequential/gru/while/gru_cell/strided_slice_3StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_3:value:0<sequential/gru/while/gru_cell/strided_slice_3/stack:output:0>sequential/gru/while/gru_cell/strided_slice_3/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2/
-sequential/gru/while/gru_cell/strided_slice_3?
%sequential/gru/while/gru_cell/BiasAddBiasAdd.sequential/gru/while/gru_cell/MatMul:product:06sequential/gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2'
%sequential/gru/while/gru_cell/BiasAdd?
.sequential/gru/while/gru_cell/ReadVariableOp_4ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_4?
3sequential/gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/gru/while/gru_cell/strided_slice_4/stack?
5sequential/gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:027
5sequential/gru/while/gru_cell/strided_slice_4/stack_1?
5sequential/gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_4/stack_2?
-sequential/gru/while/gru_cell/strided_slice_4StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_4:value:0<sequential/gru/while/gru_cell/strided_slice_4/stack:output:0>sequential/gru/while/gru_cell/strided_slice_4/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2/
-sequential/gru/while/gru_cell/strided_slice_4?
'sequential/gru/while/gru_cell/BiasAdd_1BiasAdd0sequential/gru/while/gru_cell/MatMul_1:product:06sequential/gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2)
'sequential/gru/while/gru_cell/BiasAdd_1?
.sequential/gru/while/gru_cell/ReadVariableOp_5ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_5?
3sequential/gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:025
3sequential/gru/while/gru_cell/strided_slice_5/stack?
5sequential/gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5sequential/gru/while/gru_cell/strided_slice_5/stack_1?
5sequential/gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_5/stack_2?
-sequential/gru/while/gru_cell/strided_slice_5StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_5:value:0<sequential/gru/while/gru_cell/strided_slice_5/stack:output:0>sequential/gru/while/gru_cell/strided_slice_5/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_5?
'sequential/gru/while/gru_cell/BiasAdd_2BiasAdd0sequential/gru/while/gru_cell/MatMul_2:product:06sequential/gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2)
'sequential/gru/while/gru_cell/BiasAdd_2?
.sequential/gru/while/gru_cell/ReadVariableOp_6ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_6?
3sequential/gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential/gru/while/gru_cell/strided_slice_6/stack?
5sequential/gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5sequential/gru/while/gru_cell/strided_slice_6/stack_1?
5sequential/gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_6/stack_2?
-sequential/gru/while/gru_cell/strided_slice_6StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_6:value:0<sequential/gru/while/gru_cell/strided_slice_6/stack:output:0>sequential/gru/while/gru_cell/strided_slice_6/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_6?
&sequential/gru/while/gru_cell/MatMul_3MatMul"sequential_gru_while_placeholder_26sequential/gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/gru/while/gru_cell/MatMul_3?
.sequential/gru/while/gru_cell/ReadVariableOp_7ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_7?
3sequential/gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       25
3sequential/gru/while/gru_cell/strided_slice_7/stack?
5sequential/gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   27
5sequential/gru/while/gru_cell/strided_slice_7/stack_1?
5sequential/gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_7/stack_2?
-sequential/gru/while/gru_cell/strided_slice_7StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_7:value:0<sequential/gru/while/gru_cell/strided_slice_7/stack:output:0>sequential/gru/while/gru_cell/strided_slice_7/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_7?
&sequential/gru/while/gru_cell/MatMul_4MatMul"sequential_gru_while_placeholder_26sequential/gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/gru/while/gru_cell/MatMul_4?
!sequential/gru/while/gru_cell/addAddV2.sequential/gru/while/gru_cell/BiasAdd:output:00sequential/gru/while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2#
!sequential/gru/while/gru_cell/add?
#sequential/gru/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2%
#sequential/gru/while/gru_cell/Const?
%sequential/gru/while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%sequential/gru/while/gru_cell/Const_1?
!sequential/gru/while/gru_cell/MulMul%sequential/gru/while/gru_cell/add:z:0,sequential/gru/while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2#
!sequential/gru/while/gru_cell/Mul?
#sequential/gru/while/gru_cell/Add_1AddV2%sequential/gru/while/gru_cell/Mul:z:0.sequential/gru/while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/Add_1?
5sequential/gru/while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5sequential/gru/while/gru_cell/clip_by_value/Minimum/y?
3sequential/gru/while/gru_cell/clip_by_value/MinimumMinimum'sequential/gru/while/gru_cell/Add_1:z:0>sequential/gru/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????25
3sequential/gru/while/gru_cell/clip_by_value/Minimum?
-sequential/gru/while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential/gru/while/gru_cell/clip_by_value/y?
+sequential/gru/while/gru_cell/clip_by_valueMaximum7sequential/gru/while/gru_cell/clip_by_value/Minimum:z:06sequential/gru/while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2-
+sequential/gru/while/gru_cell/clip_by_value?
#sequential/gru/while/gru_cell/add_2AddV20sequential/gru/while/gru_cell/BiasAdd_1:output:00sequential/gru/while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/add_2?
%sequential/gru/while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%sequential/gru/while/gru_cell/Const_2?
%sequential/gru/while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%sequential/gru/while/gru_cell/Const_3?
#sequential/gru/while/gru_cell/Mul_1Mul'sequential/gru/while/gru_cell/add_2:z:0.sequential/gru/while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/Mul_1?
#sequential/gru/while/gru_cell/Add_3AddV2'sequential/gru/while/gru_cell/Mul_1:z:0.sequential/gru/while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/Add_3?
7sequential/gru/while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??29
7sequential/gru/while/gru_cell/clip_by_value_1/Minimum/y?
5sequential/gru/while/gru_cell/clip_by_value_1/MinimumMinimum'sequential/gru/while/gru_cell/Add_3:z:0@sequential/gru/while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????27
5sequential/gru/while/gru_cell/clip_by_value_1/Minimum?
/sequential/gru/while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential/gru/while/gru_cell/clip_by_value_1/y?
-sequential/gru/while/gru_cell/clip_by_value_1Maximum9sequential/gru/while/gru_cell/clip_by_value_1/Minimum:z:08sequential/gru/while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2/
-sequential/gru/while/gru_cell/clip_by_value_1?
#sequential/gru/while/gru_cell/mul_2Mul1sequential/gru/while/gru_cell/clip_by_value_1:z:0"sequential_gru_while_placeholder_2*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/mul_2?
.sequential/gru/while/gru_cell/ReadVariableOp_8ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_8?
3sequential/gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   25
3sequential/gru/while/gru_cell/strided_slice_8/stack?
5sequential/gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5sequential/gru/while/gru_cell/strided_slice_8/stack_1?
5sequential/gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_8/stack_2?
-sequential/gru/while/gru_cell/strided_slice_8StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_8:value:0<sequential/gru/while/gru_cell/strided_slice_8/stack:output:0>sequential/gru/while/gru_cell/strided_slice_8/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_8?
&sequential/gru/while/gru_cell/MatMul_5MatMul'sequential/gru/while/gru_cell/mul_2:z:06sequential/gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/gru/while/gru_cell/MatMul_5?
#sequential/gru/while/gru_cell/add_4AddV20sequential/gru/while/gru_cell/BiasAdd_2:output:00sequential/gru/while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/add_4?
"sequential/gru/while/gru_cell/ReluRelu'sequential/gru/while/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2$
"sequential/gru/while/gru_cell/Relu?
#sequential/gru/while/gru_cell/mul_3Mul/sequential/gru/while/gru_cell/clip_by_value:z:0"sequential_gru_while_placeholder_2*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/mul_3?
#sequential/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential/gru/while/gru_cell/sub/x?
!sequential/gru/while/gru_cell/subSub,sequential/gru/while/gru_cell/sub/x:output:0/sequential/gru/while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2#
!sequential/gru/while/gru_cell/sub?
#sequential/gru/while/gru_cell/mul_4Mul%sequential/gru/while/gru_cell/sub:z:00sequential/gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/mul_4?
#sequential/gru/while/gru_cell/add_5AddV2'sequential/gru/while/gru_cell/mul_3:z:0'sequential/gru/while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2%
#sequential/gru/while/gru_cell/add_5?
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"sequential_gru_while_placeholder_1 sequential_gru_while_placeholder'sequential/gru/while/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype02;
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemz
sequential/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/while/add/y?
sequential/gru/while/addAddV2 sequential_gru_while_placeholder#sequential/gru/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/while/add~
sequential/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/while/add_1/y?
sequential/gru/while/add_1AddV26sequential_gru_while_sequential_gru_while_loop_counter%sequential/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/while/add_1?
sequential/gru/while/IdentityIdentitysequential/gru/while/add_1:z:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: 2
sequential/gru/while/Identity?
sequential/gru/while/Identity_1Identity<sequential_gru_while_sequential_gru_while_maximum_iterations^sequential/gru/while/NoOp*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_1?
sequential/gru/while/Identity_2Identitysequential/gru/while/add:z:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_2?
sequential/gru/while/Identity_3IdentityIsequential/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_3?
sequential/gru/while/Identity_4Identity'sequential/gru/while/gru_cell/add_5:z:0^sequential/gru/while/NoOp*
T0*'
_output_shapes
:?????????2!
sequential/gru/while/Identity_4?
sequential/gru/while/NoOpNoOp-^sequential/gru/while/gru_cell/ReadVariableOp/^sequential/gru/while/gru_cell/ReadVariableOp_1/^sequential/gru/while/gru_cell/ReadVariableOp_2/^sequential/gru/while/gru_cell/ReadVariableOp_3/^sequential/gru/while/gru_cell/ReadVariableOp_4/^sequential/gru/while/gru_cell/ReadVariableOp_5/^sequential/gru/while/gru_cell/ReadVariableOp_6/^sequential/gru/while/gru_cell/ReadVariableOp_7/^sequential/gru/while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
sequential/gru/while/NoOp"t
7sequential_gru_while_gru_cell_readvariableop_3_resource9sequential_gru_while_gru_cell_readvariableop_3_resource_0"t
7sequential_gru_while_gru_cell_readvariableop_6_resource9sequential_gru_while_gru_cell_readvariableop_6_resource_0"p
5sequential_gru_while_gru_cell_readvariableop_resource7sequential_gru_while_gru_cell_readvariableop_resource_0"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0"K
sequential_gru_while_identity_1(sequential/gru/while/Identity_1:output:0"K
sequential_gru_while_identity_2(sequential/gru/while/Identity_2:output:0"K
sequential_gru_while_identity_3(sequential/gru/while/Identity_3:output:0"K
sequential_gru_while_identity_4(sequential/gru/while/Identity_4:output:0"l
3sequential_gru_while_sequential_gru_strided_slice_15sequential_gru_while_sequential_gru_strided_slice_1_0"?
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2\
,sequential/gru/while/gru_cell/ReadVariableOp,sequential/gru/while/gru_cell/ReadVariableOp2`
.sequential/gru/while/gru_cell/ReadVariableOp_1.sequential/gru/while/gru_cell/ReadVariableOp_12`
.sequential/gru/while/gru_cell/ReadVariableOp_2.sequential/gru/while/gru_cell/ReadVariableOp_22`
.sequential/gru/while/gru_cell/ReadVariableOp_3.sequential/gru/while/gru_cell/ReadVariableOp_32`
.sequential/gru/while/gru_cell/ReadVariableOp_4.sequential/gru/while/gru_cell/ReadVariableOp_42`
.sequential/gru/while/gru_cell/ReadVariableOp_5.sequential/gru/while/gru_cell/ReadVariableOp_52`
.sequential/gru/while/gru_cell/ReadVariableOp_6.sequential/gru/while/gru_cell/ReadVariableOp_62`
.sequential/gru/while/gru_cell/ReadVariableOp_7.sequential/gru/while/gru_cell/ReadVariableOp_72`
.sequential/gru/while/gru_cell/ReadVariableOp_8.sequential/gru/while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?	
while_body_33410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?H8
*while_gru_cell_readvariableop_3_resource_0:H<
*while_gru_cell_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?H6
(while_gru_cell_readvariableop_3_resource:H:
(while_gru_cell_readvariableop_6_resource:H??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?while/gru_cell/ReadVariableOp_7?while/gru_cell/ReadVariableOp_8?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02
while/gru_cell/ReadVariableOp?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_1?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_2ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_2?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_6?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_6:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile_placeholder_2'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_7ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_7?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_7:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile_placeholder_2'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/addq
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Constu
while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_1?
while/gru_cell/MulMulwhile/gru_cell/add:z:0while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul?
while/gru_cell/Add_1AddV2while/gru_cell/Mul:z:0while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_1?
&while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&while/gru_cell/clip_by_value/Minimum/y?
$while/gru_cell/clip_by_value/MinimumMinimumwhile/gru_cell/Add_1:z:0/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$while/gru_cell/clip_by_value/Minimum?
while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
while/gru_cell/clip_by_value/y?
while/gru_cell/clip_by_valueMaximum(while/gru_cell/clip_by_value/Minimum:z:0'while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/clip_by_value?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_2u
while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Const_2u
while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_3?
while/gru_cell/Mul_1Mulwhile/gru_cell/add_2:z:0while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul_1?
while/gru_cell/Add_3AddV2while/gru_cell/Mul_1:z:0while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_3?
(while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(while/gru_cell/clip_by_value_1/Minimum/y?
&while/gru_cell/clip_by_value_1/MinimumMinimumwhile/gru_cell/Add_3:z:01while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/gru_cell/clip_by_value_1/Minimum?
 while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 while/gru_cell/clip_by_value_1/y?
while/gru_cell/clip_by_value_1Maximum*while/gru_cell/clip_by_value_1/Minimum:z:0)while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2 
while/gru_cell/clip_by_value_1?
while/gru_cell/mul_2Mul"while/gru_cell/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_8ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_8?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlice'while/gru_cell/ReadVariableOp_8:value:0-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_5?
while/gru_cell/add_4AddV2!while/gru_cell/BiasAdd_2:output:0!while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_4~
while/gru_cell/ReluReluwhile/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Relu?
while/gru_cell/mul_3Mul while/gru_cell/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_3q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0 while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/sub?
while/gru_cell/mul_4Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_4?
while/gru_cell/add_5AddV2while/gru_cell/mul_3:z:0while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6 ^while/gru_cell/ReadVariableOp_7 ^while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"V
(while_gru_cell_readvariableop_3_resource*while_gru_cell_readvariableop_3_resource_0"V
(while_gru_cell_readvariableop_6_resource*while_gru_cell_readvariableop_6_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_62B
while/gru_cell/ReadVariableOp_7while/gru_cell/ReadVariableOp_72B
while/gru_cell/ReadVariableOp_8while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_conv1d_layer_call_fn_32948

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_312062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_34568

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_309292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_31528

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
E__inference_sequential_layer_call_and_return_conditional_losses_32158
conv1d_input"
conv1d_32122: 
conv1d_32124: $
conv1d_1_32127: @
conv1d_1_32129:@
	gru_32135:	?H
	gru_32137:H
	gru_32139:H(
time_distributed_32143: $
time_distributed_32145: *
time_distributed_1_32150: &
time_distributed_1_32152:
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?gru/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_32122conv1d_32124*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_312062 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_32127conv1d_1_32129*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_312282"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_312412
max_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_312492
flatten/PartitionedCall?
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_repeat_vector_layer_call_and_return_conditional_losses_312582
repeat_vector/PartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	gru_32135	gru_32137	gru_32139*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_315152
gru/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_315282
dropout/PartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0time_distributed_32143time_distributed_32145*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_315432*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape dropout/PartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_32150time_distributed_1_32152*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_315642,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape?
IdentityIdentity3time_distributed_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
?
?
0__inference_time_distributed_layer_call_fn_34228

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_315432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34258

inputs8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_31079

inputs
dense_1_31069: 
dense_1_31071:
identity??dense_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_31069dense_1_31071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_310682!
dense_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?1
?
E__inference_sequential_layer_call_and_return_conditional_losses_32067

inputs"
conv1d_32031: 
conv1d_32033: $
conv1d_1_32036: @
conv1d_1_32038:@
	gru_32044:	?H
	gru_32046:H
	gru_32048:H(
time_distributed_32052: $
time_distributed_32054: *
time_distributed_1_32059: &
time_distributed_1_32061:
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?gru/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_32031conv1d_32033*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_312062 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_32036conv1d_1_32038*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_312282"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_312412
max_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_312492
flatten/PartitionedCall?
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_repeat_vector_layer_call_and_return_conditional_losses_312582
repeat_vector/PartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	gru_32044	gru_32046	gru_32048*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_319542
gru/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_316822!
dropout/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0time_distributed_32052time_distributed_32054*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_316552*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape(dropout/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_32059time_distributed_1_32061*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_316232,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape?
IdentityIdentity3time_distributed_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?	
while_body_33154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?H8
*while_gru_cell_readvariableop_3_resource_0:H<
*while_gru_cell_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?H6
(while_gru_cell_readvariableop_3_resource:H:
(while_gru_cell_readvariableop_6_resource:H??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?while/gru_cell/ReadVariableOp_7?while/gru_cell/ReadVariableOp_8?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02
while/gru_cell/ReadVariableOp?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_1?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_2ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_2?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_6?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_6:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile_placeholder_2'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_7ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_7?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_7:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile_placeholder_2'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/addq
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Constu
while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_1?
while/gru_cell/MulMulwhile/gru_cell/add:z:0while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul?
while/gru_cell/Add_1AddV2while/gru_cell/Mul:z:0while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_1?
&while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&while/gru_cell/clip_by_value/Minimum/y?
$while/gru_cell/clip_by_value/MinimumMinimumwhile/gru_cell/Add_1:z:0/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$while/gru_cell/clip_by_value/Minimum?
while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
while/gru_cell/clip_by_value/y?
while/gru_cell/clip_by_valueMaximum(while/gru_cell/clip_by_value/Minimum:z:0'while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/clip_by_value?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_2u
while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Const_2u
while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_3?
while/gru_cell/Mul_1Mulwhile/gru_cell/add_2:z:0while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul_1?
while/gru_cell/Add_3AddV2while/gru_cell/Mul_1:z:0while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_3?
(while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(while/gru_cell/clip_by_value_1/Minimum/y?
&while/gru_cell/clip_by_value_1/MinimumMinimumwhile/gru_cell/Add_3:z:01while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/gru_cell/clip_by_value_1/Minimum?
 while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 while/gru_cell/clip_by_value_1/y?
while/gru_cell/clip_by_value_1Maximum*while/gru_cell/clip_by_value_1/Minimum:z:0)while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2 
while/gru_cell/clip_by_value_1?
while/gru_cell/mul_2Mul"while/gru_cell/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_8ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_8?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlice'while/gru_cell/ReadVariableOp_8:value:0-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_5?
while/gru_cell/add_4AddV2!while/gru_cell/BiasAdd_2:output:0!while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_4~
while/gru_cell/ReluReluwhile/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Relu?
while/gru_cell/mul_3Mul while/gru_cell/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_3q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0 while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/sub?
while/gru_cell/mul_4Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_4?
while/gru_cell/add_5AddV2while/gru_cell/mul_3:z:0while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6 ^while/gru_cell/ReadVariableOp_7 ^while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"V
(while_gru_cell_readvariableop_3_resource*while_gru_cell_readvariableop_3_resource_0"V
(while_gru_cell_readvariableop_6_resource*while_gru_cell_readvariableop_6_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_62B
while/gru_cell/ReadVariableOp_7while/gru_cell/ReadVariableOp_72B
while/gru_cell/ReadVariableOp_8while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_31376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_31376___redundant_placeholder03
/while_while_cond_31376___redundant_placeholder13
/while_while_cond_31376___redundant_placeholder23
/while_while_cond_31376___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
I
-__inference_repeat_vector_layer_call_fn_33036

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_repeat_vector_layer_call_and_return_conditional_losses_312582
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?	
while_body_33922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?H8
*while_gru_cell_readvariableop_3_resource_0:H<
*while_gru_cell_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?H6
(while_gru_cell_readvariableop_3_resource:H:
(while_gru_cell_readvariableop_6_resource:H??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?while/gru_cell/ReadVariableOp_7?while/gru_cell/ReadVariableOp_8?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02
while/gru_cell/ReadVariableOp?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_1?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_2ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_2?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_6?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_6:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile_placeholder_2'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_7ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_7?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_7:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile_placeholder_2'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/addq
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Constu
while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_1?
while/gru_cell/MulMulwhile/gru_cell/add:z:0while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul?
while/gru_cell/Add_1AddV2while/gru_cell/Mul:z:0while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_1?
&while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&while/gru_cell/clip_by_value/Minimum/y?
$while/gru_cell/clip_by_value/MinimumMinimumwhile/gru_cell/Add_1:z:0/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$while/gru_cell/clip_by_value/Minimum?
while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
while/gru_cell/clip_by_value/y?
while/gru_cell/clip_by_valueMaximum(while/gru_cell/clip_by_value/Minimum:z:0'while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/clip_by_value?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_2u
while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Const_2u
while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_3?
while/gru_cell/Mul_1Mulwhile/gru_cell/add_2:z:0while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul_1?
while/gru_cell/Add_3AddV2while/gru_cell/Mul_1:z:0while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_3?
(while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(while/gru_cell/clip_by_value_1/Minimum/y?
&while/gru_cell/clip_by_value_1/MinimumMinimumwhile/gru_cell/Add_3:z:01while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/gru_cell/clip_by_value_1/Minimum?
 while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 while/gru_cell/clip_by_value_1/y?
while/gru_cell/clip_by_value_1Maximum*while/gru_cell/clip_by_value_1/Minimum:z:0)while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2 
while/gru_cell/clip_by_value_1?
while/gru_cell/mul_2Mul"while/gru_cell/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_8ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_8?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlice'while/gru_cell/ReadVariableOp_8:value:0-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_5?
while/gru_cell/add_4AddV2!while/gru_cell/BiasAdd_2:output:0!while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_4~
while/gru_cell/ReluReluwhile/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Relu?
while/gru_cell/mul_3Mul while/gru_cell/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_3q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0 while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/sub?
while/gru_cell/mul_4Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_4?
while/gru_cell/add_5AddV2while/gru_cell/mul_3:z:0while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6 ^while/gru_cell/ReadVariableOp_7 ^while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"V
(while_gru_cell_readvariableop_3_resource*while_gru_cell_readvariableop_3_resource_0"V
(while_gru_cell_readvariableop_6_resource*while_gru_cell_readvariableop_6_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_62B
while/gru_cell/ReadVariableOp_7while/gru_cell/ReadVariableOp_72B
while/gru_cell/ReadVariableOp_8while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_31241

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
(__inference_gru_cell_layer_call_fn_34535

inputs
states_0
unknown:	?H
	unknown_0:H
	unknown_1:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_303512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:??????????:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_31655

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_time_distributed_layer_call_fn_34210

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_309402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_31068

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
2__inference_time_distributed_1_layer_call_fn_34325

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_311272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
??
?	
while_body_31816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?H8
*while_gru_cell_readvariableop_3_resource_0:H<
*while_gru_cell_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?H6
(while_gru_cell_readvariableop_3_resource:H:
(while_gru_cell_readvariableop_6_resource:H??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?while/gru_cell/ReadVariableOp_7?while/gru_cell/ReadVariableOp_8?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02
while/gru_cell/ReadVariableOp?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_1?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_2ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_2?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_6?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_6:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile_placeholder_2'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_7ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_7?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_7:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile_placeholder_2'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/addq
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Constu
while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_1?
while/gru_cell/MulMulwhile/gru_cell/add:z:0while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul?
while/gru_cell/Add_1AddV2while/gru_cell/Mul:z:0while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_1?
&while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&while/gru_cell/clip_by_value/Minimum/y?
$while/gru_cell/clip_by_value/MinimumMinimumwhile/gru_cell/Add_1:z:0/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$while/gru_cell/clip_by_value/Minimum?
while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
while/gru_cell/clip_by_value/y?
while/gru_cell/clip_by_valueMaximum(while/gru_cell/clip_by_value/Minimum:z:0'while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/clip_by_value?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_2u
while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Const_2u
while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_3?
while/gru_cell/Mul_1Mulwhile/gru_cell/add_2:z:0while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul_1?
while/gru_cell/Add_3AddV2while/gru_cell/Mul_1:z:0while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_3?
(while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(while/gru_cell/clip_by_value_1/Minimum/y?
&while/gru_cell/clip_by_value_1/MinimumMinimumwhile/gru_cell/Add_3:z:01while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/gru_cell/clip_by_value_1/Minimum?
 while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 while/gru_cell/clip_by_value_1/y?
while/gru_cell/clip_by_value_1Maximum*while/gru_cell/clip_by_value_1/Minimum:z:0)while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2 
while/gru_cell/clip_by_value_1?
while/gru_cell/mul_2Mul"while/gru_cell/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_8ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_8?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlice'while/gru_cell/ReadVariableOp_8:value:0-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_5?
while/gru_cell/add_4AddV2!while/gru_cell/BiasAdd_2:output:0!while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_4~
while/gru_cell/ReluReluwhile/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Relu?
while/gru_cell/mul_3Mul while/gru_cell/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_3q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0 while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/sub?
while/gru_cell/mul_4Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_4?
while/gru_cell/add_5AddV2while/gru_cell/mul_3:z:0while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6 ^while/gru_cell/ReadVariableOp_7 ^while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"V
(while_gru_cell_readvariableop_3_resource*while_gru_cell_readvariableop_3_resource_0"V
(while_gru_cell_readvariableop_6_resource*while_gru_cell_readvariableop_6_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_62B
while/gru_cell/ReadVariableOp_7while/gru_cell/ReadVariableOp_72B
while/gru_cell/ReadVariableOp_8while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
sequential_gru_while_cond_30007:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_2<
8sequential_gru_while_less_sequential_gru_strided_slice_1Q
Msequential_gru_while_sequential_gru_while_cond_30007___redundant_placeholder0Q
Msequential_gru_while_sequential_gru_while_cond_30007___redundant_placeholder1Q
Msequential_gru_while_sequential_gru_while_cond_30007___redundant_placeholder2Q
Msequential_gru_while_sequential_gru_while_cond_30007___redundant_placeholder3!
sequential_gru_while_identity
?
sequential/gru/while/LessLess sequential_gru_while_placeholder8sequential_gru_while_less_sequential_gru_strided_slice_1*
T0*
_output_shapes
: 2
sequential/gru/while/Less?
sequential/gru/while/IdentityIdentitysequential/gru/while/Less:z:0*
T0
*
_output_shapes
: 2
sequential/gru/while/Identity"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
I
-__inference_max_pooling1d_layer_call_fn_32999

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_312412
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_34578

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_30171
conv1d_inputS
=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource: ?
1sequential_conv1d_biasadd_readvariableop_resource: U
?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource: @A
3sequential_conv1d_1_biasadd_readvariableop_resource:@B
/sequential_gru_gru_cell_readvariableop_resource:	?H?
1sequential_gru_gru_cell_readvariableop_3_resource:HC
1sequential_gru_gru_cell_readvariableop_6_resource:HR
@sequential_time_distributed_dense_matmul_readvariableop_resource: O
Asequential_time_distributed_dense_biasadd_readvariableop_resource: V
Dsequential_time_distributed_1_dense_1_matmul_readvariableop_resource: S
Esequential_time_distributed_1_dense_1_biasadd_readvariableop_resource:
identity??(sequential/conv1d/BiasAdd/ReadVariableOp?4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp?*sequential/conv1d_1/BiasAdd/ReadVariableOp?6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?&sequential/gru/gru_cell/ReadVariableOp?(sequential/gru/gru_cell/ReadVariableOp_1?(sequential/gru/gru_cell/ReadVariableOp_2?(sequential/gru/gru_cell/ReadVariableOp_3?(sequential/gru/gru_cell/ReadVariableOp_4?(sequential/gru/gru_cell/ReadVariableOp_5?(sequential/gru/gru_cell/ReadVariableOp_6?(sequential/gru/gru_cell/ReadVariableOp_7?(sequential/gru/gru_cell/ReadVariableOp_8?sequential/gru/while?8sequential/time_distributed/dense/BiasAdd/ReadVariableOp?7sequential/time_distributed/dense/MatMul/ReadVariableOp?<sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp?;sequential/time_distributed_1/dense_1/MatMul/ReadVariableOp?
'sequential/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'sequential/conv1d/conv1d/ExpandDims/dim?
#sequential/conv1d/conv1d/ExpandDims
ExpandDimsconv1d_input0sequential/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2%
#sequential/conv1d/conv1d/ExpandDims?
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype026
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
)sequential/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/conv1d/conv1d/ExpandDims_1/dim?
%sequential/conv1d/conv1d/ExpandDims_1
ExpandDims<sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2'
%sequential/conv1d/conv1d/ExpandDims_1?
sequential/conv1d/conv1dConv2D,sequential/conv1d/conv1d/ExpandDims:output:0.sequential/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential/conv1d/conv1d?
 sequential/conv1d/conv1d/SqueezeSqueeze!sequential/conv1d/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2"
 sequential/conv1d/conv1d/Squeeze?
(sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv1d/BiasAdd/ReadVariableOp?
sequential/conv1d/BiasAddBiasAdd)sequential/conv1d/conv1d/Squeeze:output:00sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
sequential/conv1d/BiasAdd?
sequential/conv1d/ReluRelu"sequential/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
sequential/conv1d/Relu?
)sequential/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential/conv1d_1/conv1d/ExpandDims/dim?
%sequential/conv1d_1/conv1d/ExpandDims
ExpandDims$sequential/conv1d/Relu:activations:02sequential/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2'
%sequential/conv1d_1/conv1d/ExpandDims?
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype028
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
+sequential/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_1/conv1d/ExpandDims_1/dim?
'sequential/conv1d_1/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2)
'sequential/conv1d_1/conv1d/ExpandDims_1?
sequential/conv1d_1/conv1dConv2D.sequential/conv1d_1/conv1d/ExpandDims:output:00sequential/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential/conv1d_1/conv1d?
"sequential/conv1d_1/conv1d/SqueezeSqueeze#sequential/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2$
"sequential/conv1d_1/conv1d/Squeeze?
*sequential/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv1d_1/BiasAdd/ReadVariableOp?
sequential/conv1d_1/BiasAddBiasAdd+sequential/conv1d_1/conv1d/Squeeze:output:02sequential/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
sequential/conv1d_1/BiasAdd?
sequential/conv1d_1/ReluRelu$sequential/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
sequential/conv1d_1/Relu?
'sequential/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/max_pooling1d/ExpandDims/dim?
#sequential/max_pooling1d/ExpandDims
ExpandDims&sequential/conv1d_1/Relu:activations:00sequential/max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2%
#sequential/max_pooling1d/ExpandDims?
 sequential/max_pooling1d/MaxPoolMaxPool,sequential/max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling1d/MaxPool?
 sequential/max_pooling1d/SqueezeSqueeze)sequential/max_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2"
 sequential/max_pooling1d/Squeeze?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape)sequential/max_pooling1d/Squeeze:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
'sequential/repeat_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/repeat_vector/ExpandDims/dim?
#sequential/repeat_vector/ExpandDims
ExpandDims#sequential/flatten/Reshape:output:00sequential/repeat_vector/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2%
#sequential/repeat_vector/ExpandDims?
sequential/repeat_vector/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2 
sequential/repeat_vector/stack?
sequential/repeat_vector/TileTile,sequential/repeat_vector/ExpandDims:output:0'sequential/repeat_vector/stack:output:0*
T0*,
_output_shapes
:??????????2
sequential/repeat_vector/Tile?
sequential/gru/ShapeShape&sequential/repeat_vector/Tile:output:0*
T0*
_output_shapes
:2
sequential/gru/Shape?
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/gru/strided_slice/stack?
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_1?
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_2?
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/gru/strided_slicez
sequential/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/zeros/mul/y?
sequential/gru/zeros/mulMul%sequential/gru/strided_slice:output:0#sequential/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/mul}
sequential/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
sequential/gru/zeros/Less/y?
sequential/gru/zeros/LessLesssequential/gru/zeros/mul:z:0$sequential/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/Less?
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/zeros/packed/1?
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/gru/zeros/packed}
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/zeros/Const?
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential/gru/zeros?
sequential/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
sequential/gru/transpose/perm?
sequential/gru/transpose	Transpose&sequential/repeat_vector/Tile:output:0&sequential/gru/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
sequential/gru/transpose|
sequential/gru/Shape_1Shapesequential/gru/transpose:y:0*
T0*
_output_shapes
:2
sequential/gru/Shape_1?
$sequential/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/gru/strided_slice_1/stack?
&sequential/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_1/stack_1?
&sequential/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_1/stack_2?
sequential/gru/strided_slice_1StridedSlicesequential/gru/Shape_1:output:0-sequential/gru/strided_slice_1/stack:output:0/sequential/gru/strided_slice_1/stack_1:output:0/sequential/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential/gru/strided_slice_1?
*sequential/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*sequential/gru/TensorArrayV2/element_shape?
sequential/gru/TensorArrayV2TensorListReserve3sequential/gru/TensorArrayV2/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/gru/TensorArrayV2?
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
6sequential/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru/transpose:y:0Msequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6sequential/gru/TensorArrayUnstack/TensorListFromTensor?
$sequential/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/gru/strided_slice_2/stack?
&sequential/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_2/stack_1?
&sequential/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_2/stack_2?
sequential/gru/strided_slice_2StridedSlicesequential/gru/transpose:y:0-sequential/gru/strided_slice_2/stack:output:0/sequential/gru/strided_slice_2/stack_1:output:0/sequential/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2 
sequential/gru/strided_slice_2?
&sequential/gru/gru_cell/ReadVariableOpReadVariableOp/sequential_gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02(
&sequential/gru/gru_cell/ReadVariableOp?
+sequential/gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+sequential/gru/gru_cell/strided_slice/stack?
-sequential/gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-sequential/gru/gru_cell/strided_slice/stack_1?
-sequential/gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-sequential/gru/gru_cell/strided_slice/stack_2?
%sequential/gru/gru_cell/strided_sliceStridedSlice.sequential/gru/gru_cell/ReadVariableOp:value:04sequential/gru/gru_cell/strided_slice/stack:output:06sequential/gru/gru_cell/strided_slice/stack_1:output:06sequential/gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2'
%sequential/gru/gru_cell/strided_slice?
sequential/gru/gru_cell/MatMulMatMul'sequential/gru/strided_slice_2:output:0.sequential/gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2 
sequential/gru/gru_cell/MatMul?
(sequential/gru/gru_cell/ReadVariableOp_1ReadVariableOp/sequential_gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_1?
-sequential/gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-sequential/gru/gru_cell/strided_slice_1/stack?
/sequential/gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   21
/sequential/gru/gru_cell/strided_slice_1/stack_1?
/sequential/gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_1/stack_2?
'sequential/gru/gru_cell/strided_slice_1StridedSlice0sequential/gru/gru_cell/ReadVariableOp_1:value:06sequential/gru/gru_cell/strided_slice_1/stack:output:08sequential/gru/gru_cell/strided_slice_1/stack_1:output:08sequential/gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_1?
 sequential/gru/gru_cell/MatMul_1MatMul'sequential/gru/strided_slice_2:output:00sequential/gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2"
 sequential/gru/gru_cell/MatMul_1?
(sequential/gru/gru_cell/ReadVariableOp_2ReadVariableOp/sequential_gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_2?
-sequential/gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2/
-sequential/gru/gru_cell/strided_slice_2/stack?
/sequential/gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/sequential/gru/gru_cell/strided_slice_2/stack_1?
/sequential/gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_2/stack_2?
'sequential/gru/gru_cell/strided_slice_2StridedSlice0sequential/gru/gru_cell/ReadVariableOp_2:value:06sequential/gru/gru_cell/strided_slice_2/stack:output:08sequential/gru/gru_cell/strided_slice_2/stack_1:output:08sequential/gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_2?
 sequential/gru/gru_cell/MatMul_2MatMul'sequential/gru/strided_slice_2:output:00sequential/gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2"
 sequential/gru/gru_cell/MatMul_2?
(sequential/gru/gru_cell/ReadVariableOp_3ReadVariableOp1sequential_gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_3?
-sequential/gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/gru/gru_cell/strided_slice_3/stack?
/sequential/gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_3/stack_1?
/sequential/gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_3/stack_2?
'sequential/gru/gru_cell/strided_slice_3StridedSlice0sequential/gru/gru_cell/ReadVariableOp_3:value:06sequential/gru/gru_cell/strided_slice_3/stack:output:08sequential/gru/gru_cell/strided_slice_3/stack_1:output:08sequential/gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'sequential/gru/gru_cell/strided_slice_3?
sequential/gru/gru_cell/BiasAddBiasAdd(sequential/gru/gru_cell/MatMul:product:00sequential/gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2!
sequential/gru/gru_cell/BiasAdd?
(sequential/gru/gru_cell/ReadVariableOp_4ReadVariableOp1sequential_gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_4?
-sequential/gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/gru/gru_cell/strided_slice_4/stack?
/sequential/gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:021
/sequential/gru/gru_cell/strided_slice_4/stack_1?
/sequential/gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_4/stack_2?
'sequential/gru/gru_cell/strided_slice_4StridedSlice0sequential/gru/gru_cell/ReadVariableOp_4:value:06sequential/gru/gru_cell/strided_slice_4/stack:output:08sequential/gru/gru_cell/strided_slice_4/stack_1:output:08sequential/gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'sequential/gru/gru_cell/strided_slice_4?
!sequential/gru/gru_cell/BiasAdd_1BiasAdd*sequential/gru/gru_cell/MatMul_1:product:00sequential/gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2#
!sequential/gru/gru_cell/BiasAdd_1?
(sequential/gru/gru_cell/ReadVariableOp_5ReadVariableOp1sequential_gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_5?
-sequential/gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02/
-sequential/gru/gru_cell/strided_slice_5/stack?
/sequential/gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/sequential/gru/gru_cell/strided_slice_5/stack_1?
/sequential/gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_5/stack_2?
'sequential/gru/gru_cell/strided_slice_5StridedSlice0sequential/gru/gru_cell/ReadVariableOp_5:value:06sequential/gru/gru_cell/strided_slice_5/stack:output:08sequential/gru/gru_cell/strided_slice_5/stack_1:output:08sequential/gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2)
'sequential/gru/gru_cell/strided_slice_5?
!sequential/gru/gru_cell/BiasAdd_2BiasAdd*sequential/gru/gru_cell/MatMul_2:product:00sequential/gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2#
!sequential/gru/gru_cell/BiasAdd_2?
(sequential/gru/gru_cell/ReadVariableOp_6ReadVariableOp1sequential_gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_6?
-sequential/gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-sequential/gru/gru_cell/strided_slice_6/stack?
/sequential/gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/sequential/gru/gru_cell/strided_slice_6/stack_1?
/sequential/gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_6/stack_2?
'sequential/gru/gru_cell/strided_slice_6StridedSlice0sequential/gru/gru_cell/ReadVariableOp_6:value:06sequential/gru/gru_cell/strided_slice_6/stack:output:08sequential/gru/gru_cell/strided_slice_6/stack_1:output:08sequential/gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_6?
 sequential/gru/gru_cell/MatMul_3MatMulsequential/gru/zeros:output:00sequential/gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2"
 sequential/gru/gru_cell/MatMul_3?
(sequential/gru/gru_cell/ReadVariableOp_7ReadVariableOp1sequential_gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_7?
-sequential/gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2/
-sequential/gru/gru_cell/strided_slice_7/stack?
/sequential/gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   21
/sequential/gru/gru_cell/strided_slice_7/stack_1?
/sequential/gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_7/stack_2?
'sequential/gru/gru_cell/strided_slice_7StridedSlice0sequential/gru/gru_cell/ReadVariableOp_7:value:06sequential/gru/gru_cell/strided_slice_7/stack:output:08sequential/gru/gru_cell/strided_slice_7/stack_1:output:08sequential/gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_7?
 sequential/gru/gru_cell/MatMul_4MatMulsequential/gru/zeros:output:00sequential/gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2"
 sequential/gru/gru_cell/MatMul_4?
sequential/gru/gru_cell/addAddV2(sequential/gru/gru_cell/BiasAdd:output:0*sequential/gru/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/add?
sequential/gru/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
sequential/gru/gru_cell/Const?
sequential/gru/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential/gru/gru_cell/Const_1?
sequential/gru/gru_cell/MulMulsequential/gru/gru_cell/add:z:0&sequential/gru/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/Mul?
sequential/gru/gru_cell/Add_1AddV2sequential/gru/gru_cell/Mul:z:0(sequential/gru/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/Add_1?
/sequential/gru/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??21
/sequential/gru/gru_cell/clip_by_value/Minimum/y?
-sequential/gru/gru_cell/clip_by_value/MinimumMinimum!sequential/gru/gru_cell/Add_1:z:08sequential/gru/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2/
-sequential/gru/gru_cell/clip_by_value/Minimum?
'sequential/gru/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'sequential/gru/gru_cell/clip_by_value/y?
%sequential/gru/gru_cell/clip_by_valueMaximum1sequential/gru/gru_cell/clip_by_value/Minimum:z:00sequential/gru/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2'
%sequential/gru/gru_cell/clip_by_value?
sequential/gru/gru_cell/add_2AddV2*sequential/gru/gru_cell/BiasAdd_1:output:0*sequential/gru/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/add_2?
sequential/gru/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
sequential/gru/gru_cell/Const_2?
sequential/gru/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential/gru/gru_cell/Const_3?
sequential/gru/gru_cell/Mul_1Mul!sequential/gru/gru_cell/add_2:z:0(sequential/gru/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/Mul_1?
sequential/gru/gru_cell/Add_3AddV2!sequential/gru/gru_cell/Mul_1:z:0(sequential/gru/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/Add_3?
1sequential/gru/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1sequential/gru/gru_cell/clip_by_value_1/Minimum/y?
/sequential/gru/gru_cell/clip_by_value_1/MinimumMinimum!sequential/gru/gru_cell/Add_3:z:0:sequential/gru/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????21
/sequential/gru/gru_cell/clip_by_value_1/Minimum?
)sequential/gru/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)sequential/gru/gru_cell/clip_by_value_1/y?
'sequential/gru/gru_cell/clip_by_value_1Maximum3sequential/gru/gru_cell/clip_by_value_1/Minimum:z:02sequential/gru/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2)
'sequential/gru/gru_cell/clip_by_value_1?
sequential/gru/gru_cell/mul_2Mul+sequential/gru/gru_cell/clip_by_value_1:z:0sequential/gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/mul_2?
(sequential/gru/gru_cell/ReadVariableOp_8ReadVariableOp1sequential_gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_8?
-sequential/gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2/
-sequential/gru/gru_cell/strided_slice_8/stack?
/sequential/gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/sequential/gru/gru_cell/strided_slice_8/stack_1?
/sequential/gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_8/stack_2?
'sequential/gru/gru_cell/strided_slice_8StridedSlice0sequential/gru/gru_cell/ReadVariableOp_8:value:06sequential/gru/gru_cell/strided_slice_8/stack:output:08sequential/gru/gru_cell/strided_slice_8/stack_1:output:08sequential/gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_8?
 sequential/gru/gru_cell/MatMul_5MatMul!sequential/gru/gru_cell/mul_2:z:00sequential/gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2"
 sequential/gru/gru_cell/MatMul_5?
sequential/gru/gru_cell/add_4AddV2*sequential/gru/gru_cell/BiasAdd_2:output:0*sequential/gru/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/add_4?
sequential/gru/gru_cell/ReluRelu!sequential/gru/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/Relu?
sequential/gru/gru_cell/mul_3Mul)sequential/gru/gru_cell/clip_by_value:z:0sequential/gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/mul_3?
sequential/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/gru/gru_cell/sub/x?
sequential/gru/gru_cell/subSub&sequential/gru/gru_cell/sub/x:output:0)sequential/gru/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/sub?
sequential/gru/gru_cell/mul_4Mulsequential/gru/gru_cell/sub:z:0*sequential/gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/mul_4?
sequential/gru/gru_cell/add_5AddV2!sequential/gru/gru_cell/mul_3:z:0!sequential/gru/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
sequential/gru/gru_cell/add_5?
,sequential/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2.
,sequential/gru/TensorArrayV2_1/element_shape?
sequential/gru/TensorArrayV2_1TensorListReserve5sequential/gru/TensorArrayV2_1/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential/gru/TensorArrayV2_1l
sequential/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru/time?
'sequential/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'sequential/gru/while/maximum_iterations?
!sequential/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/gru/while/loop_counter?
sequential/gru/whileWhile*sequential/gru/while/loop_counter:output:00sequential/gru/while/maximum_iterations:output:0sequential/gru/time:output:0'sequential/gru/TensorArrayV2_1:handle:0sequential/gru/zeros:output:0'sequential/gru/strided_slice_1:output:0Fsequential/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/sequential_gru_gru_cell_readvariableop_resource1sequential_gru_gru_cell_readvariableop_3_resource1sequential_gru_gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
sequential_gru_while_body_30008*+
cond#R!
sequential_gru_while_cond_30007*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
sequential/gru/while?
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shape?
1sequential/gru/TensorArrayV2Stack/TensorListStackTensorListStacksequential/gru/while:output:3Hsequential/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype023
1sequential/gru/TensorArrayV2Stack/TensorListStack?
$sequential/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$sequential/gru/strided_slice_3/stack?
&sequential/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/gru/strided_slice_3/stack_1?
&sequential/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_3/stack_2?
sequential/gru/strided_slice_3StridedSlice:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0-sequential/gru/strided_slice_3/stack:output:0/sequential/gru/strided_slice_3/stack_1:output:0/sequential/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2 
sequential/gru/strided_slice_3?
sequential/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/gru/transpose_1/perm?
sequential/gru/transpose_1	Transpose:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
sequential/gru/transpose_1?
sequential/dropout/IdentityIdentitysequential/gru/transpose_1:y:0*
T0*+
_output_shapes
:?????????2
sequential/dropout/Identity?
)sequential/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)sequential/time_distributed/Reshape/shape?
#sequential/time_distributed/ReshapeReshape$sequential/dropout/Identity:output:02sequential/time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2%
#sequential/time_distributed/Reshape?
7sequential/time_distributed/dense/MatMul/ReadVariableOpReadVariableOp@sequential_time_distributed_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype029
7sequential/time_distributed/dense/MatMul/ReadVariableOp?
(sequential/time_distributed/dense/MatMulMatMul,sequential/time_distributed/Reshape:output:0?sequential/time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential/time_distributed/dense/MatMul?
8sequential/time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOpAsequential_time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8sequential/time_distributed/dense/BiasAdd/ReadVariableOp?
)sequential/time_distributed/dense/BiasAddBiasAdd2sequential/time_distributed/dense/MatMul:product:0@sequential/time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2+
)sequential/time_distributed/dense/BiasAdd?
+sequential/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2-
+sequential/time_distributed/Reshape_1/shape?
%sequential/time_distributed/Reshape_1Reshape2sequential/time_distributed/dense/BiasAdd:output:04sequential/time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2'
%sequential/time_distributed/Reshape_1?
+sequential/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2-
+sequential/time_distributed/Reshape_2/shape?
%sequential/time_distributed/Reshape_2Reshape$sequential/dropout/Identity:output:04sequential/time_distributed/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2'
%sequential/time_distributed/Reshape_2?
+sequential/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2-
+sequential/time_distributed_1/Reshape/shape?
%sequential/time_distributed_1/ReshapeReshape.sequential/time_distributed/Reshape_1:output:04sequential/time_distributed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2'
%sequential/time_distributed_1/Reshape?
;sequential/time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOpDsequential_time_distributed_1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02=
;sequential/time_distributed_1/dense_1/MatMul/ReadVariableOp?
,sequential/time_distributed_1/dense_1/MatMulMatMul.sequential/time_distributed_1/Reshape:output:0Csequential/time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,sequential/time_distributed_1/dense_1/MatMul?
<sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpEsequential_time_distributed_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp?
-sequential/time_distributed_1/dense_1/BiasAddBiasAdd6sequential/time_distributed_1/dense_1/MatMul:product:0Dsequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-sequential/time_distributed_1/dense_1/BiasAdd?
-sequential/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2/
-sequential/time_distributed_1/Reshape_1/shape?
'sequential/time_distributed_1/Reshape_1Reshape6sequential/time_distributed_1/dense_1/BiasAdd:output:06sequential/time_distributed_1/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2)
'sequential/time_distributed_1/Reshape_1?
-sequential/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2/
-sequential/time_distributed_1/Reshape_2/shape?
'sequential/time_distributed_1/Reshape_2Reshape.sequential/time_distributed/Reshape_1:output:06sequential/time_distributed_1/Reshape_2/shape:output:0*
T0*'
_output_shapes
:????????? 2)
'sequential/time_distributed_1/Reshape_2?
IdentityIdentity0sequential/time_distributed_1/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp)^sequential/conv1d/BiasAdd/ReadVariableOp5^sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp+^sequential/conv1d_1/BiasAdd/ReadVariableOp7^sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp'^sequential/gru/gru_cell/ReadVariableOp)^sequential/gru/gru_cell/ReadVariableOp_1)^sequential/gru/gru_cell/ReadVariableOp_2)^sequential/gru/gru_cell/ReadVariableOp_3)^sequential/gru/gru_cell/ReadVariableOp_4)^sequential/gru/gru_cell/ReadVariableOp_5)^sequential/gru/gru_cell/ReadVariableOp_6)^sequential/gru/gru_cell/ReadVariableOp_7)^sequential/gru/gru_cell/ReadVariableOp_8^sequential/gru/while9^sequential/time_distributed/dense/BiasAdd/ReadVariableOp8^sequential/time_distributed/dense/MatMul/ReadVariableOp=^sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp<^sequential/time_distributed_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2T
(sequential/conv1d/BiasAdd/ReadVariableOp(sequential/conv1d/BiasAdd/ReadVariableOp2l
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential/conv1d_1/BiasAdd/ReadVariableOp*sequential/conv1d_1/BiasAdd/ReadVariableOp2p
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2P
&sequential/gru/gru_cell/ReadVariableOp&sequential/gru/gru_cell/ReadVariableOp2T
(sequential/gru/gru_cell/ReadVariableOp_1(sequential/gru/gru_cell/ReadVariableOp_12T
(sequential/gru/gru_cell/ReadVariableOp_2(sequential/gru/gru_cell/ReadVariableOp_22T
(sequential/gru/gru_cell/ReadVariableOp_3(sequential/gru/gru_cell/ReadVariableOp_32T
(sequential/gru/gru_cell/ReadVariableOp_4(sequential/gru/gru_cell/ReadVariableOp_42T
(sequential/gru/gru_cell/ReadVariableOp_5(sequential/gru/gru_cell/ReadVariableOp_52T
(sequential/gru/gru_cell/ReadVariableOp_6(sequential/gru/gru_cell/ReadVariableOp_62T
(sequential/gru/gru_cell/ReadVariableOp_7(sequential/gru/gru_cell/ReadVariableOp_72T
(sequential/gru/gru_cell/ReadVariableOp_8(sequential/gru/gru_cell/ReadVariableOp_82,
sequential/gru/whilesequential/gru/while2t
8sequential/time_distributed/dense/BiasAdd/ReadVariableOp8sequential/time_distributed/dense/BiasAdd/ReadVariableOp2r
7sequential/time_distributed/dense/MatMul/ReadVariableOp7sequential/time_distributed/dense/MatMul/ReadVariableOp2|
<sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp<sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp2z
;sequential/time_distributed_1/dense_1/MatMul/ReadVariableOp;sequential/time_distributed_1/dense_1/MatMul/ReadVariableOp:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
?!
?
while_body_30364
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
while_gru_cell_30386_0:	?H$
while_gru_cell_30388_0:H(
while_gru_cell_30390_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
while_gru_cell_30386:	?H"
while_gru_cell_30388:H&
while_gru_cell_30390:H??&while/gru_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_30386_0while_gru_cell_30388_0while_gru_cell_30390_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_303512(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp".
while_gru_cell_30386while_gru_cell_30386_0".
while_gru_cell_30388while_gru_cell_30388_0".
while_gru_cell_30390while_gru_cell_30390_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?
>__inference_gru_layer_call_and_return_conditional_losses_33292
inputs_03
 gru_cell_readvariableop_resource:	?H0
"gru_cell_readvariableop_3_resource:H4
"gru_cell_readvariableop_6_resource:H
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?gru_cell/ReadVariableOp_7?gru_cell/ReadVariableOp_8?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_2ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_2?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_1?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_2?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_6:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulzeros:output:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_7ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_7?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_7:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulzeros:output:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/adde
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Consti
gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_1?
gru_cell/MulMulgru_cell/add:z:0gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul?
gru_cell/Add_1AddV2gru_cell/Mul:z:0gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_1?
 gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_cell/clip_by_value/Minimum/y?
gru_cell/clip_by_value/MinimumMinimumgru_cell/Add_1:z:0)gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2 
gru_cell/clip_by_value/Minimumy
gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value/y?
gru_cell/clip_by_valueMaximum"gru_cell/clip_by_value/Minimum:z:0!gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value?
gru_cell/add_2AddV2gru_cell/BiasAdd_1:output:0gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_2i
gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Const_2i
gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_3?
gru_cell/Mul_1Mulgru_cell/add_2:z:0gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul_1?
gru_cell/Add_3AddV2gru_cell/Mul_1:z:0gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_3?
"gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru_cell/clip_by_value_1/Minimum/y?
 gru_cell/clip_by_value_1/MinimumMinimumgru_cell/Add_3:z:0+gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_cell/clip_by_value_1/Minimum}
gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value_1/y?
gru_cell/clip_by_value_1Maximum$gru_cell/clip_by_value_1/Minimum:z:0#gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value_1?
gru_cell/mul_2Mulgru_cell/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_8ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_8?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlice!gru_cell/ReadVariableOp_8:value:0'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_8?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_5?
gru_cell/add_4AddV2gru_cell/BiasAdd_2:output:0gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_4l
gru_cell/ReluRelugru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/Relu?
gru_cell/mul_3Mulgru_cell/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_3e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/sub?
gru_cell/mul_4Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_4?
gru_cell/add_5AddV2gru_cell/mul_3:z:0gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_3_resource"gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_33154*
condR
while_cond_33153*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1w
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^gru_cell/ReadVariableOp_7^gru_cell/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_626
gru_cell/ReadVariableOp_7gru_cell/ReadVariableOp_726
gru_cell/ReadVariableOp_8gru_cell/ReadVariableOp_82
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
??
?
>__inference_gru_layer_call_and_return_conditional_losses_31954

inputs3
 gru_cell_readvariableop_resource:	?H0
"gru_cell_readvariableop_3_resource:H4
"gru_cell_readvariableop_6_resource:H
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?gru_cell/ReadVariableOp_7?gru_cell/ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_2ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_2?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_1?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_2?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_6:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulzeros:output:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_7ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_7?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_7:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulzeros:output:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/adde
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Consti
gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_1?
gru_cell/MulMulgru_cell/add:z:0gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul?
gru_cell/Add_1AddV2gru_cell/Mul:z:0gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_1?
 gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_cell/clip_by_value/Minimum/y?
gru_cell/clip_by_value/MinimumMinimumgru_cell/Add_1:z:0)gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2 
gru_cell/clip_by_value/Minimumy
gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value/y?
gru_cell/clip_by_valueMaximum"gru_cell/clip_by_value/Minimum:z:0!gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value?
gru_cell/add_2AddV2gru_cell/BiasAdd_1:output:0gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_2i
gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Const_2i
gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_3?
gru_cell/Mul_1Mulgru_cell/add_2:z:0gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul_1?
gru_cell/Add_3AddV2gru_cell/Mul_1:z:0gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_3?
"gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru_cell/clip_by_value_1/Minimum/y?
 gru_cell/clip_by_value_1/MinimumMinimumgru_cell/Add_3:z:0+gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_cell/clip_by_value_1/Minimum}
gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value_1/y?
gru_cell/clip_by_value_1Maximum$gru_cell/clip_by_value_1/Minimum:z:0#gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value_1?
gru_cell/mul_2Mulgru_cell/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_8ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_8?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlice!gru_cell/ReadVariableOp_8:value:0'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_8?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_5?
gru_cell/add_4AddV2gru_cell/BiasAdd_2:output:0gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_4l
gru_cell/ReluRelugru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/Relu?
gru_cell/mul_3Mulgru_cell/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_3e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/sub?
gru_cell/mul_4Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_4?
gru_cell/add_5AddV2gru_cell/mul_3:z:0gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_3_resource"gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_31816*
condR
while_cond_31815*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1n
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^gru_cell/ReadVariableOp_7^gru_cell/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_626
gru_cell/ReadVariableOp_7gru_cell/ReadVariableOp_726
gru_cell/ReadVariableOp_8gru_cell/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
Ռ
?
E__inference_sequential_layer_call_and_return_conditional_losses_32547

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_1_biasadd_readvariableop_resource:@7
$gru_gru_cell_readvariableop_resource:	?H4
&gru_gru_cell_readvariableop_3_resource:H8
&gru_gru_cell_readvariableop_6_resource:HG
5time_distributed_dense_matmul_readvariableop_resource: D
6time_distributed_dense_biasadd_readvariableop_resource: K
9time_distributed_1_dense_1_matmul_readvariableop_resource: H
:time_distributed_1_dense_1_biasadd_readvariableop_resource:
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?gru/gru_cell/ReadVariableOp?gru/gru_cell/ReadVariableOp_1?gru/gru_cell/ReadVariableOp_2?gru/gru_cell/ReadVariableOp_3?gru/gru_cell/ReadVariableOp_4?gru/gru_cell/ReadVariableOp_5?gru/gru_cell/ReadVariableOp_6?gru/gru_cell/ReadVariableOp_7?gru/gru_cell/ReadVariableOp_8?	gru/while?-time_distributed/dense/BiasAdd/ReadVariableOp?,time_distributed/dense/MatMul/ReadVariableOp?1time_distributed_1/dense_1/BiasAdd/ReadVariableOp?0time_distributed_1/dense_1/MatMul/ReadVariableOp?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
conv1d/Relu?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
max_pooling1d/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapemax_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape~
repeat_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
repeat_vector/ExpandDims/dim?
repeat_vector/ExpandDims
ExpandDimsflatten/Reshape:output:0%repeat_vector/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector/ExpandDims
repeat_vector/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
repeat_vector/stack?
repeat_vector/TileTile!repeat_vector/ExpandDims:output:0repeat_vector/stack:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector/Tilea
	gru/ShapeShaperepeat_vector/Tile:output:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transposerepeat_vector/Tile:output:0gru/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru/gru_cell/ReadVariableOp?
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack?
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"gru/gru_cell/strided_slice/stack_1?
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2?
gru/gru_cell/strided_sliceStridedSlice#gru/gru_cell/ReadVariableOp:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice?
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul?
gru/gru_cell/ReadVariableOp_1ReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru/gru_cell/ReadVariableOp_1?
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"gru/gru_cell/strided_slice_1/stack?
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2&
$gru/gru_cell/strided_slice_1/stack_1?
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2?
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_1:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1?
gru/gru_cell/MatMul_1MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/ReadVariableOp_2ReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru/gru_cell/ReadVariableOp_2?
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2$
"gru/gru_cell/strided_slice_2/stack?
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1?
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2?
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2?
gru/gru_cell/MatMul_2MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_2?
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru/gru_cell/ReadVariableOp_3?
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack?
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_1?
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2?
gru/gru_cell/strided_slice_3StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru/gru_cell/strided_slice_3?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/BiasAdd?
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru/gru_cell/ReadVariableOp_4?
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"gru/gru_cell/strided_slice_4/stack?
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02&
$gru/gru_cell/strided_slice_4/stack_1?
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2?
gru/gru_cell/strided_slice_4StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru/gru_cell/strided_slice_4?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/BiasAdd_1?
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru/gru_cell/ReadVariableOp_5?
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02$
"gru/gru_cell/strided_slice_5/stack?
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1?
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2?
gru/gru_cell/strided_slice_5StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru/gru_cell/strided_slice_5?
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/BiasAdd_2?
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru/gru_cell/ReadVariableOp_6?
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack?
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$gru/gru_cell/strided_slice_6/stack_1?
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2?
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6?
gru/gru_cell/MatMul_3MatMulgru/zeros:output:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_3?
gru/gru_cell/ReadVariableOp_7ReadVariableOp&gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru/gru_cell/ReadVariableOp_7?
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"gru/gru_cell/strided_slice_7/stack?
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2&
$gru/gru_cell/strided_slice_7/stack_1?
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2?
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_7:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7?
gru/gru_cell/MatMul_4MatMulgru/zeros:output:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_4?
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/addm
gru/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/gru_cell/Constq
gru/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/gru_cell/Const_1?
gru/gru_cell/MulMulgru/gru_cell/add:z:0gru/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Mul?
gru/gru_cell/Add_1AddV2gru/gru_cell/Mul:z:0gru/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Add_1?
$gru/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$gru/gru_cell/clip_by_value/Minimum/y?
"gru/gru_cell/clip_by_value/MinimumMinimumgru/gru_cell/Add_1:z:0-gru/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2$
"gru/gru_cell/clip_by_value/Minimum?
gru/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/gru_cell/clip_by_value/y?
gru/gru_cell/clip_by_valueMaximum&gru/gru_cell/clip_by_value/Minimum:z:0%gru/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/clip_by_value?
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/add_2q
gru/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/gru_cell/Const_2q
gru/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/gru_cell/Const_3?
gru/gru_cell/Mul_1Mulgru/gru_cell/add_2:z:0gru/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Mul_1?
gru/gru_cell/Add_3AddV2gru/gru_cell/Mul_1:z:0gru/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Add_3?
&gru/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gru/gru_cell/clip_by_value_1/Minimum/y?
$gru/gru_cell/clip_by_value_1/MinimumMinimumgru/gru_cell/Add_3:z:0/gru/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$gru/gru_cell/clip_by_value_1/Minimum?
gru/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
gru/gru_cell/clip_by_value_1/y?
gru/gru_cell/clip_by_value_1Maximum(gru/gru_cell/clip_by_value_1/Minimum:z:0'gru/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/clip_by_value_1?
gru/gru_cell/mul_2Mul gru/gru_cell/clip_by_value_1:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/mul_2?
gru/gru_cell/ReadVariableOp_8ReadVariableOp&gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru/gru_cell/ReadVariableOp_8?
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2$
"gru/gru_cell/strided_slice_8/stack?
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_8/stack_1?
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_8/stack_2?
gru/gru_cell/strided_slice_8StridedSlice%gru/gru_cell/ReadVariableOp_8:value:0+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_8?
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_5?
gru/gru_cell/add_4AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/add_4x
gru/gru_cell/ReluRelugru/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Relu?
gru/gru_cell/mul_3Mulgru/gru_cell/clip_by_value:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/mul_3m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/sub?
gru/gru_cell/mul_4Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/mul_4?
gru/gru_cell/add_5AddV2gru/gru_cell/mul_3:z:0gru/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/add_5?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_3_resource&gru_gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( * 
bodyR
gru_while_body_32384* 
condR
gru_while_cond_32383*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru/strided_slice_3?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
gru/transpose_1{
dropout/IdentityIdentitygru/transpose_1:y:0*
T0*+
_output_shapes
:?????????2
dropout/Identity?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapedropout/Identity:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape?
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,time_distributed/dense/MatMul/ReadVariableOp?
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
time_distributed/dense/MatMul?
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-time_distributed/dense/BiasAdd/ReadVariableOp?
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
time_distributed/dense/BiasAdd?
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed/Reshape_1/shape?
time_distributed/Reshape_1Reshape'time_distributed/dense/BiasAdd:output:0)time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed/Reshape_1?
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2"
 time_distributed/Reshape_2/shape?
time_distributed/Reshape_2Reshapedropout/Identity:output:0)time_distributed/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape_2?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape?
0time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOp9time_distributed_1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0time_distributed_1/dense_1/MatMul/ReadVariableOp?
!time_distributed_1/dense_1/MatMulMatMul#time_distributed_1/Reshape:output:08time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!time_distributed_1/dense_1/MatMul?
1time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp?
"time_distributed_1/dense_1/BiasAddBiasAdd+time_distributed_1/dense_1/MatMul:product:09time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"time_distributed_1/dense_1/BiasAdd?
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2$
"time_distributed_1/Reshape_1/shape?
time_distributed_1/Reshape_1Reshape+time_distributed_1/dense_1/BiasAdd:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
time_distributed_1/Reshape_1?
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2$
"time_distributed_1/Reshape_2/shape?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape_2?
IdentityIdentity%time_distributed_1/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_6^gru/gru_cell/ReadVariableOp_7^gru/gru_cell/ReadVariableOp_8
^gru/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp2^time_distributed_1/dense_1/BiasAdd/ReadVariableOp1^time_distributed_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62>
gru/gru_cell/ReadVariableOp_7gru/gru_cell/ReadVariableOp_72>
gru/gru_cell/ReadVariableOp_8gru/gru_cell/ReadVariableOp_82
	gru/while	gru/while2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp2f
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp1time_distributed_1/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_1/MatMul/ReadVariableOp0time_distributed_1/dense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
>__inference_gru_layer_call_and_return_conditional_losses_34060

inputs3
 gru_cell_readvariableop_resource:	?H0
"gru_cell_readvariableop_3_resource:H4
"gru_cell_readvariableop_6_resource:H
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?gru_cell/ReadVariableOp_7?gru_cell/ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_2ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_2?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_1?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_2?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_6:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulzeros:output:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_7ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_7?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_7:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulzeros:output:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/adde
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Consti
gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_1?
gru_cell/MulMulgru_cell/add:z:0gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul?
gru_cell/Add_1AddV2gru_cell/Mul:z:0gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_1?
 gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_cell/clip_by_value/Minimum/y?
gru_cell/clip_by_value/MinimumMinimumgru_cell/Add_1:z:0)gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2 
gru_cell/clip_by_value/Minimumy
gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value/y?
gru_cell/clip_by_valueMaximum"gru_cell/clip_by_value/Minimum:z:0!gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value?
gru_cell/add_2AddV2gru_cell/BiasAdd_1:output:0gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_2i
gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Const_2i
gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_3?
gru_cell/Mul_1Mulgru_cell/add_2:z:0gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul_1?
gru_cell/Add_3AddV2gru_cell/Mul_1:z:0gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_3?
"gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru_cell/clip_by_value_1/Minimum/y?
 gru_cell/clip_by_value_1/MinimumMinimumgru_cell/Add_3:z:0+gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_cell/clip_by_value_1/Minimum}
gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value_1/y?
gru_cell/clip_by_value_1Maximum$gru_cell/clip_by_value_1/Minimum:z:0#gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value_1?
gru_cell/mul_2Mulgru_cell/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_8ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_8?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlice!gru_cell/ReadVariableOp_8:value:0'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_8?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_5?
gru_cell/add_4AddV2gru_cell/BiasAdd_2:output:0gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_4l
gru_cell/ReluRelugru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/Relu?
gru_cell/mul_3Mulgru_cell/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_3e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/sub?
gru_cell/mul_4Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_4?
gru_cell/add_5AddV2gru_cell/mul_3:z:0gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_3_resource"gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_33922*
condR
while_cond_33921*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1n
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^gru_cell/ReadVariableOp_7^gru_cell/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_626
gru_cell/ReadVariableOp_7gru_cell/ReadVariableOp_726
gru_cell/ReadVariableOp_8gru_cell/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_33005

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34279

inputs8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
ɧ
?

gru_while_body_32384$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	?H<
.gru_while_gru_cell_readvariableop_3_resource_0:H@
.gru_while_gru_cell_readvariableop_6_resource_0:H
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	?H:
,gru_while_gru_cell_readvariableop_3_resource:H>
,gru_while_gru_cell_readvariableop_6_resource:H??!gru/while/gru_cell/ReadVariableOp?#gru/while/gru_cell/ReadVariableOp_1?#gru/while/gru_cell/ReadVariableOp_2?#gru/while/gru_cell/ReadVariableOp_3?#gru/while/gru_cell/ReadVariableOp_4?#gru/while/gru_cell/ReadVariableOp_5?#gru/while/gru_cell/ReadVariableOp_6?#gru/while/gru_cell/ReadVariableOp_7?#gru/while/gru_cell/ReadVariableOp_8?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack?
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(gru/while/gru_cell/strided_slice/stack_1?
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2?
 gru/while/gru_cell/strided_sliceStridedSlice)gru/while/gru_cell/ReadVariableOp:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul?
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1?
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru/while/gru_cell/strided_slice_1/stack?
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru/while/gru_cell/strided_slice_1/stack_1?
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2?
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1?
gru/while/gru_cell/MatMul_1MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_1?
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2?
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru/while/gru_cell/strided_slice_2/stack?
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1?
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2?
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2?
gru/while/gru_cell/MatMul_2MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_2?
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3?
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack?
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_1?
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2?
"gru/while/gru_cell/strided_slice_3StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"gru/while/gru_cell/strided_slice_3?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/BiasAdd?
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4?
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(gru/while/gru_cell/strided_slice_4/stack?
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02,
*gru/while/gru_cell/strided_slice_4/stack_1?
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2?
"gru/while/gru_cell/strided_slice_4StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/strided_slice_4?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/BiasAdd_1?
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5?
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02*
(gru/while/gru_cell/strided_slice_5/stack?
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1?
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2?
"gru/while/gru_cell/strided_slice_5StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2$
"gru/while/gru_cell/strided_slice_5?
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/BiasAdd_2?
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6?
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack?
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*gru/while/gru_cell/strided_slice_6/stack_1?
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2?
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6?
gru/while/gru_cell/MatMul_3MatMulgru_while_placeholder_2+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_3?
#gru/while/gru_cell/ReadVariableOp_7ReadVariableOp.gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_7?
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru/while/gru_cell/strided_slice_7/stack?
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru/while/gru_cell/strided_slice_7/stack_1?
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2?
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_7:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7?
gru/while/gru_cell/MatMul_4MatMulgru_while_placeholder_2+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_4?
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/addy
gru/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/while/gru_cell/Const}
gru/while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/while/gru_cell/Const_1?
gru/while/gru_cell/MulMulgru/while/gru_cell/add:z:0!gru/while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Mul?
gru/while/gru_cell/Add_1AddV2gru/while/gru_cell/Mul:z:0#gru/while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Add_1?
*gru/while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*gru/while/gru_cell/clip_by_value/Minimum/y?
(gru/while/gru_cell/clip_by_value/MinimumMinimumgru/while/gru_cell/Add_1:z:03gru/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(gru/while/gru_cell/clip_by_value/Minimum?
"gru/while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"gru/while/gru_cell/clip_by_value/y?
 gru/while/gru_cell/clip_by_valueMaximum,gru/while/gru_cell/clip_by_value/Minimum:z:0+gru/while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru/while/gru_cell/clip_by_value?
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/add_2}
gru/while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/while/gru_cell/Const_2}
gru/while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/while/gru_cell/Const_3?
gru/while/gru_cell/Mul_1Mulgru/while/gru_cell/add_2:z:0#gru/while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Mul_1?
gru/while/gru_cell/Add_3AddV2gru/while/gru_cell/Mul_1:z:0#gru/while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Add_3?
,gru/while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,gru/while/gru_cell/clip_by_value_1/Minimum/y?
*gru/while/gru_cell/clip_by_value_1/MinimumMinimumgru/while/gru_cell/Add_3:z:05gru/while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2,
*gru/while/gru_cell/clip_by_value_1/Minimum?
$gru/while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$gru/while/gru_cell/clip_by_value_1/y?
"gru/while/gru_cell/clip_by_value_1Maximum.gru/while/gru_cell/clip_by_value_1/Minimum:z:0-gru/while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2$
"gru/while/gru_cell/clip_by_value_1?
gru/while/gru_cell/mul_2Mul&gru/while/gru_cell/clip_by_value_1:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/mul_2?
#gru/while/gru_cell/ReadVariableOp_8ReadVariableOp.gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_8?
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru/while/gru_cell/strided_slice_8/stack?
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_8/stack_1?
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_8/stack_2?
"gru/while/gru_cell/strided_slice_8StridedSlice+gru/while/gru_cell/ReadVariableOp_8:value:01gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_8?
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_5?
gru/while/gru_cell/add_4AddV2%gru/while/gru_cell/BiasAdd_2:output:0%gru/while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/add_4?
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Relu?
gru/while/gru_cell/mul_3Mul$gru/while/gru_cell/clip_by_value:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/mul_3y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0$gru/while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_4Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/mul_4?
gru/while/gru_cell/add_5AddV2gru/while/gru_cell/mul_3:z:0gru/while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/add_5?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1{
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity_1}
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_5:z:0^gru/while/NoOp*
T0*'
_output_shapes
:?????????2
gru/while/Identity_4?
gru/while/NoOpNoOp"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6$^gru/while/gru_cell/ReadVariableOp_7$^gru/while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
gru/while/NoOp"^
,gru_while_gru_cell_readvariableop_3_resource.gru_while_gru_cell_readvariableop_3_resource_0"^
,gru_while_gru_cell_readvariableop_6_resource.gru_while_gru_cell_readvariableop_6_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_62J
#gru/while/gru_cell/ReadVariableOp_7#gru/while/gru_cell/ReadVariableOp_72J
#gru/while/gru_cell/ReadVariableOp_8#gru/while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_32989

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_34082
inputs_0
unknown:	?H
	unknown_0:H
	unknown_1:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_306732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_30363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_30363___redundant_placeholder03
/while_while_cond_30363___redundant_placeholder13
/while_while_cond_30363___redundant_placeholder23
/while_while_cond_30363___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34307

inputs8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
0__inference_time_distributed_layer_call_fn_34237

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_316552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv1d_1_layer_call_fn_32973

inputs
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_312282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_34093

inputs
unknown:	?H
	unknown_0:H
	unknown_1:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_315152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_time_distributed_1_layer_call_fn_34316

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_310792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
??
?
>__inference_gru_layer_call_and_return_conditional_losses_33548
inputs_03
 gru_cell_readvariableop_resource:	?H0
"gru_cell_readvariableop_3_resource:H4
"gru_cell_readvariableop_6_resource:H
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?gru_cell/ReadVariableOp_7?gru_cell/ReadVariableOp_8?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_2ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_2?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_1?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_2?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_6:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulzeros:output:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_7ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_7?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_7:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulzeros:output:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/adde
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Consti
gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_1?
gru_cell/MulMulgru_cell/add:z:0gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul?
gru_cell/Add_1AddV2gru_cell/Mul:z:0gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_1?
 gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_cell/clip_by_value/Minimum/y?
gru_cell/clip_by_value/MinimumMinimumgru_cell/Add_1:z:0)gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2 
gru_cell/clip_by_value/Minimumy
gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value/y?
gru_cell/clip_by_valueMaximum"gru_cell/clip_by_value/Minimum:z:0!gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value?
gru_cell/add_2AddV2gru_cell/BiasAdd_1:output:0gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_2i
gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Const_2i
gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_3?
gru_cell/Mul_1Mulgru_cell/add_2:z:0gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul_1?
gru_cell/Add_3AddV2gru_cell/Mul_1:z:0gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_3?
"gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru_cell/clip_by_value_1/Minimum/y?
 gru_cell/clip_by_value_1/MinimumMinimumgru_cell/Add_3:z:0+gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_cell/clip_by_value_1/Minimum}
gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value_1/y?
gru_cell/clip_by_value_1Maximum$gru_cell/clip_by_value_1/Minimum:z:0#gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value_1?
gru_cell/mul_2Mulgru_cell/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_8ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_8?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlice!gru_cell/ReadVariableOp_8:value:0'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_8?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_5?
gru_cell/add_4AddV2gru_cell/BiasAdd_2:output:0gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_4l
gru_cell/ReluRelugru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/Relu?
gru_cell/mul_3Mulgru_cell/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_3e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/sub?
gru_cell/mul_4Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_4?
gru_cell/add_5AddV2gru_cell/mul_3:z:0gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_3_resource"gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_33410*
condR
while_cond_33409*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1w
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^gru_cell/ReadVariableOp_7^gru_cell/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_626
gru_cell/ReadVariableOp_7gru_cell/ReadVariableOp_726
gru_cell/ReadVariableOp_8gru_cell/ReadVariableOp_82
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34152

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?a
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_34432

inputs
states_0*
readvariableop_resource:	?H'
readvariableop_3_resource:H+
readvariableop_6_resource:H
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddz
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1z
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3~
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7t
MatMul_4MatMulstates_0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Muld
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1f
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_2~
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:?????????2
Relud
mul_3Mulclip_by_value:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subd
mul_4Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5d
IdentityIdentity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:??????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
ޕ
?
E__inference_sequential_layer_call_and_return_conditional_losses_32869

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_1_biasadd_readvariableop_resource:@7
$gru_gru_cell_readvariableop_resource:	?H4
&gru_gru_cell_readvariableop_3_resource:H8
&gru_gru_cell_readvariableop_6_resource:HG
5time_distributed_dense_matmul_readvariableop_resource: D
6time_distributed_dense_biasadd_readvariableop_resource: K
9time_distributed_1_dense_1_matmul_readvariableop_resource: H
:time_distributed_1_dense_1_biasadd_readvariableop_resource:
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?gru/gru_cell/ReadVariableOp?gru/gru_cell/ReadVariableOp_1?gru/gru_cell/ReadVariableOp_2?gru/gru_cell/ReadVariableOp_3?gru/gru_cell/ReadVariableOp_4?gru/gru_cell/ReadVariableOp_5?gru/gru_cell/ReadVariableOp_6?gru/gru_cell/ReadVariableOp_7?gru/gru_cell/ReadVariableOp_8?	gru/while?-time_distributed/dense/BiasAdd/ReadVariableOp?,time_distributed/dense/MatMul/ReadVariableOp?1time_distributed_1/dense_1/BiasAdd/ReadVariableOp?0time_distributed_1/dense_1/MatMul/ReadVariableOp?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
conv1d/Relu?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
max_pooling1d/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapemax_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape~
repeat_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
repeat_vector/ExpandDims/dim?
repeat_vector/ExpandDims
ExpandDimsflatten/Reshape:output:0%repeat_vector/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector/ExpandDims
repeat_vector/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
repeat_vector/stack?
repeat_vector/TileTile!repeat_vector/ExpandDims:output:0repeat_vector/stack:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector/Tilea
	gru/ShapeShaperepeat_vector/Tile:output:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transposerepeat_vector/Tile:output:0gru/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru/gru_cell/ReadVariableOp?
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack?
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"gru/gru_cell/strided_slice/stack_1?
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2?
gru/gru_cell/strided_sliceStridedSlice#gru/gru_cell/ReadVariableOp:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice?
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul?
gru/gru_cell/ReadVariableOp_1ReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru/gru_cell/ReadVariableOp_1?
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"gru/gru_cell/strided_slice_1/stack?
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2&
$gru/gru_cell/strided_slice_1/stack_1?
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2?
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_1:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1?
gru/gru_cell/MatMul_1MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/ReadVariableOp_2ReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru/gru_cell/ReadVariableOp_2?
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2$
"gru/gru_cell/strided_slice_2/stack?
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1?
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2?
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2?
gru/gru_cell/MatMul_2MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_2?
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru/gru_cell/ReadVariableOp_3?
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack?
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_1?
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2?
gru/gru_cell/strided_slice_3StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru/gru_cell/strided_slice_3?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/BiasAdd?
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru/gru_cell/ReadVariableOp_4?
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"gru/gru_cell/strided_slice_4/stack?
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02&
$gru/gru_cell/strided_slice_4/stack_1?
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2?
gru/gru_cell/strided_slice_4StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru/gru_cell/strided_slice_4?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/BiasAdd_1?
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru/gru_cell/ReadVariableOp_5?
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02$
"gru/gru_cell/strided_slice_5/stack?
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1?
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2?
gru/gru_cell/strided_slice_5StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru/gru_cell/strided_slice_5?
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/BiasAdd_2?
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru/gru_cell/ReadVariableOp_6?
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack?
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$gru/gru_cell/strided_slice_6/stack_1?
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2?
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6?
gru/gru_cell/MatMul_3MatMulgru/zeros:output:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_3?
gru/gru_cell/ReadVariableOp_7ReadVariableOp&gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru/gru_cell/ReadVariableOp_7?
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"gru/gru_cell/strided_slice_7/stack?
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2&
$gru/gru_cell/strided_slice_7/stack_1?
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2?
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_7:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7?
gru/gru_cell/MatMul_4MatMulgru/zeros:output:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_4?
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/addm
gru/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/gru_cell/Constq
gru/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/gru_cell/Const_1?
gru/gru_cell/MulMulgru/gru_cell/add:z:0gru/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Mul?
gru/gru_cell/Add_1AddV2gru/gru_cell/Mul:z:0gru/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Add_1?
$gru/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$gru/gru_cell/clip_by_value/Minimum/y?
"gru/gru_cell/clip_by_value/MinimumMinimumgru/gru_cell/Add_1:z:0-gru/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2$
"gru/gru_cell/clip_by_value/Minimum?
gru/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/gru_cell/clip_by_value/y?
gru/gru_cell/clip_by_valueMaximum&gru/gru_cell/clip_by_value/Minimum:z:0%gru/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/clip_by_value?
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/add_2q
gru/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/gru_cell/Const_2q
gru/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/gru_cell/Const_3?
gru/gru_cell/Mul_1Mulgru/gru_cell/add_2:z:0gru/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Mul_1?
gru/gru_cell/Add_3AddV2gru/gru_cell/Mul_1:z:0gru/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Add_3?
&gru/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gru/gru_cell/clip_by_value_1/Minimum/y?
$gru/gru_cell/clip_by_value_1/MinimumMinimumgru/gru_cell/Add_3:z:0/gru/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$gru/gru_cell/clip_by_value_1/Minimum?
gru/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
gru/gru_cell/clip_by_value_1/y?
gru/gru_cell/clip_by_value_1Maximum(gru/gru_cell/clip_by_value_1/Minimum:z:0'gru/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/clip_by_value_1?
gru/gru_cell/mul_2Mul gru/gru_cell/clip_by_value_1:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/mul_2?
gru/gru_cell/ReadVariableOp_8ReadVariableOp&gru_gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru/gru_cell/ReadVariableOp_8?
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2$
"gru/gru_cell/strided_slice_8/stack?
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_8/stack_1?
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_8/stack_2?
gru/gru_cell/strided_slice_8StridedSlice%gru/gru_cell/ReadVariableOp_8:value:0+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_8?
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/MatMul_5?
gru/gru_cell/add_4AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/add_4x
gru/gru_cell/ReluRelugru/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/Relu?
gru/gru_cell/mul_3Mulgru/gru_cell/clip_by_value:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/mul_3m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/sub?
gru/gru_cell/mul_4Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/mul_4?
gru/gru_cell/add_5AddV2gru/gru_cell/mul_3:z:0gru/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru/gru_cell/add_5?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_3_resource&gru_gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( * 
bodyR
gru_while_body_32699* 
condR
gru_while_cond_32698*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru/strided_slice_3?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
gru/transpose_1s
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/dropout/Const?
dropout/dropout/MulMulgru/transpose_1:y:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/dropout/Mulq
dropout/dropout/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/dropout/Mul_1?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapedropout/dropout/Mul_1:z:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape?
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,time_distributed/dense/MatMul/ReadVariableOp?
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
time_distributed/dense/MatMul?
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-time_distributed/dense/BiasAdd/ReadVariableOp?
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
time_distributed/dense/BiasAdd?
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed/Reshape_1/shape?
time_distributed/Reshape_1Reshape'time_distributed/dense/BiasAdd:output:0)time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed/Reshape_1?
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2"
 time_distributed/Reshape_2/shape?
time_distributed/Reshape_2Reshapedropout/dropout/Mul_1:z:0)time_distributed/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape_2?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape?
0time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOp9time_distributed_1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0time_distributed_1/dense_1/MatMul/ReadVariableOp?
!time_distributed_1/dense_1/MatMulMatMul#time_distributed_1/Reshape:output:08time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!time_distributed_1/dense_1/MatMul?
1time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp?
"time_distributed_1/dense_1/BiasAddBiasAdd+time_distributed_1/dense_1/MatMul:product:09time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"time_distributed_1/dense_1/BiasAdd?
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2$
"time_distributed_1/Reshape_1/shape?
time_distributed_1/Reshape_1Reshape+time_distributed_1/dense_1/BiasAdd:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
time_distributed_1/Reshape_1?
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2$
"time_distributed_1/Reshape_2/shape?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape_2?
IdentityIdentity%time_distributed_1/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_6^gru/gru_cell/ReadVariableOp_7^gru/gru_cell/ReadVariableOp_8
^gru/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp2^time_distributed_1/dense_1/BiasAdd/ReadVariableOp1^time_distributed_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62>
gru/gru_cell/ReadVariableOp_7gru/gru_cell/ReadVariableOp_72>
gru/gru_cell/ReadVariableOp_8gru/gru_cell/ReadVariableOp_82
	gru/while	gru/while2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp2f
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp1time_distributed_1/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_1/MatMul/ReadVariableOp0time_distributed_1/dense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_32119
conv1d_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	?H
	unknown_4:H
	unknown_5:H
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_320672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
?
?
while_cond_31815
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_31815___redundant_placeholder03
/while_while_cond_31815___redundant_placeholder13
/while_while_cond_31815___redundant_placeholder23
/while_while_cond_31815___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?	
while_body_33666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?H8
*while_gru_cell_readvariableop_3_resource_0:H<
*while_gru_cell_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?H6
(while_gru_cell_readvariableop_3_resource:H:
(while_gru_cell_readvariableop_6_resource:H??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?while/gru_cell/ReadVariableOp_7?while/gru_cell/ReadVariableOp_8?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02
while/gru_cell/ReadVariableOp?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_1?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_2ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_2?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_6?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_6:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile_placeholder_2'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_7ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_7?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_7:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile_placeholder_2'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/addq
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Constu
while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_1?
while/gru_cell/MulMulwhile/gru_cell/add:z:0while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul?
while/gru_cell/Add_1AddV2while/gru_cell/Mul:z:0while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_1?
&while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&while/gru_cell/clip_by_value/Minimum/y?
$while/gru_cell/clip_by_value/MinimumMinimumwhile/gru_cell/Add_1:z:0/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$while/gru_cell/clip_by_value/Minimum?
while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
while/gru_cell/clip_by_value/y?
while/gru_cell/clip_by_valueMaximum(while/gru_cell/clip_by_value/Minimum:z:0'while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/clip_by_value?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_2u
while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Const_2u
while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_3?
while/gru_cell/Mul_1Mulwhile/gru_cell/add_2:z:0while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul_1?
while/gru_cell/Add_3AddV2while/gru_cell/Mul_1:z:0while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_3?
(while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(while/gru_cell/clip_by_value_1/Minimum/y?
&while/gru_cell/clip_by_value_1/MinimumMinimumwhile/gru_cell/Add_3:z:01while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/gru_cell/clip_by_value_1/Minimum?
 while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 while/gru_cell/clip_by_value_1/y?
while/gru_cell/clip_by_value_1Maximum*while/gru_cell/clip_by_value_1/Minimum:z:0)while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2 
while/gru_cell/clip_by_value_1?
while/gru_cell/mul_2Mul"while/gru_cell/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_8ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_8?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlice'while/gru_cell/ReadVariableOp_8:value:0-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_5?
while/gru_cell/add_4AddV2!while/gru_cell/BiasAdd_2:output:0!while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_4~
while/gru_cell/ReluReluwhile/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Relu?
while/gru_cell/mul_3Mul while/gru_cell/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_3q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0 while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/sub?
while/gru_cell/mul_4Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_4?
while/gru_cell/add_5AddV2while/gru_cell/mul_3:z:0while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6 ^while/gru_cell/ReadVariableOp_7 ^while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"V
(while_gru_cell_readvariableop_3_resource*while_gru_cell_readvariableop_3_resource_0"V
(while_gru_cell_readvariableop_6_resource*while_gru_cell_readvariableop_6_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_62B
while/gru_cell/ReadVariableOp_7while/gru_cell/ReadVariableOp_72B
while/gru_cell/ReadVariableOp_8while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?=
?
>__inference_gru_layer_call_and_return_conditional_losses_30673

inputs!
gru_cell_30598:	?H
gru_cell_30600:H 
gru_cell_30602:H
identity?? gru_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_30598gru_cell_30600gru_cell_30602*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_305432"
 gru_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_30598gru_cell_30600gru_cell_30602*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_30610*
condR
while_cond_30609*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1w
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityy
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_34878
file_prefix4
assignvariableop_conv1d_kernel: ,
assignvariableop_1_conv1d_bias: 8
"assignvariableop_2_conv1d_1_kernel: @.
 assignvariableop_3_conv1d_1_bias:@'
assignvariableop_4_nadam_iter:	 )
assignvariableop_5_nadam_beta_1: )
assignvariableop_6_nadam_beta_2: (
assignvariableop_7_nadam_decay: 0
&assignvariableop_8_nadam_learning_rate: 1
'assignvariableop_9_nadam_momentum_cache: :
'assignvariableop_10_gru_gru_cell_kernel:	?HC
1assignvariableop_11_gru_gru_cell_recurrent_kernel:H3
%assignvariableop_12_gru_gru_cell_bias:H=
+assignvariableop_13_time_distributed_kernel: 7
)assignvariableop_14_time_distributed_bias: ?
-assignvariableop_15_time_distributed_1_kernel: 9
+assignvariableop_16_time_distributed_1_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: ?
)assignvariableop_21_nadam_conv1d_kernel_m: 5
'assignvariableop_22_nadam_conv1d_bias_m: A
+assignvariableop_23_nadam_conv1d_1_kernel_m: @7
)assignvariableop_24_nadam_conv1d_1_bias_m:@B
/assignvariableop_25_nadam_gru_gru_cell_kernel_m:	?HK
9assignvariableop_26_nadam_gru_gru_cell_recurrent_kernel_m:H;
-assignvariableop_27_nadam_gru_gru_cell_bias_m:HE
3assignvariableop_28_nadam_time_distributed_kernel_m: ?
1assignvariableop_29_nadam_time_distributed_bias_m: G
5assignvariableop_30_nadam_time_distributed_1_kernel_m: A
3assignvariableop_31_nadam_time_distributed_1_bias_m:?
)assignvariableop_32_nadam_conv1d_kernel_v: 5
'assignvariableop_33_nadam_conv1d_bias_v: A
+assignvariableop_34_nadam_conv1d_1_kernel_v: @7
)assignvariableop_35_nadam_conv1d_1_bias_v:@B
/assignvariableop_36_nadam_gru_gru_cell_kernel_v:	?HK
9assignvariableop_37_nadam_gru_gru_cell_recurrent_kernel_v:H;
-assignvariableop_38_nadam_gru_gru_cell_bias_v:HE
3assignvariableop_39_nadam_time_distributed_kernel_v: ?
1assignvariableop_40_nadam_time_distributed_bias_v: G
5assignvariableop_41_nadam_time_distributed_1_kernel_v: A
3assignvariableop_42_nadam_time_distributed_1_bias_v:
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_nadam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_nadam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp'assignvariableop_9_nadam_momentum_cacheIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_gru_gru_cell_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_gru_gru_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_gru_gru_cell_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_time_distributed_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp)assignvariableop_14_time_distributed_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp-assignvariableop_15_time_distributed_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_time_distributed_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_nadam_conv1d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_nadam_conv1d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_nadam_conv1d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_nadam_conv1d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp/assignvariableop_25_nadam_gru_gru_cell_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp9assignvariableop_26_nadam_gru_gru_cell_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp-assignvariableop_27_nadam_gru_gru_cell_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_nadam_time_distributed_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_nadam_time_distributed_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp5assignvariableop_30_nadam_time_distributed_1_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp3assignvariableop_31_nadam_time_distributed_1_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_nadam_conv1d_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_nadam_conv1d_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_nadam_conv1d_1_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_nadam_conv1d_1_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp/assignvariableop_36_nadam_gru_gru_cell_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp9assignvariableop_37_nadam_gru_gru_cell_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp-assignvariableop_38_nadam_gru_gru_cell_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp3assignvariableop_39_nadam_time_distributed_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp1assignvariableop_40_nadam_time_distributed_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp5assignvariableop_41_nadam_time_distributed_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp3assignvariableop_42_nadam_time_distributed_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43f
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_44?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
#__inference_gru_layer_call_fn_34071
inputs_0
unknown:	?H
	unknown_0:H
	unknown_1:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_304272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
`
'__inference_dropout_layer_call_fn_34131

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_316822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_34126

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_315282
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_31228

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
H__inference_repeat_vector_layer_call_and_return_conditional_losses_33018

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
stackx
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :??????????????????2
Tilen
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_31206

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_34559

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?a
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_34521

inputs
states_0*
readvariableop_resource:	?H'
readvariableop_3_resource:H+
readvariableop_6_resource:H
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddz
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1z
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3~
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7t
MatMul_4MatMulstates_0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Muld
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1f
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_2~
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:?????????2
Relud
mul_3Mulclip_by_value:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subd
mul_4Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5d
IdentityIdentity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:??????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?
?
*__inference_sequential_layer_call_fn_32923

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	?H
	unknown_4:H
	unknown_5:H
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_320672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_33153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_33153___redundant_placeholder03
/while_while_cond_33153___redundant_placeholder13
/while_while_cond_33153___redundant_placeholder23
/while_while_cond_33153___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?!
?
while_body_30610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
while_gru_cell_30632_0:	?H$
while_gru_cell_30634_0:H(
while_gru_cell_30636_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
while_gru_cell_30632:	?H"
while_gru_cell_30634:H&
while_gru_cell_30636:H??&while/gru_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_30632_0while_gru_cell_30634_0while_gru_cell_30636_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_305432(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp".
while_gru_cell_30632while_gru_cell_30632_0".
while_gru_cell_30634while_gru_cell_30634_0".
while_gru_cell_30636while_gru_cell_30636_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?
>__inference_gru_layer_call_and_return_conditional_losses_31515

inputs3
 gru_cell_readvariableop_resource:	?H0
"gru_cell_readvariableop_3_resource:H4
"gru_cell_readvariableop_6_resource:H
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?gru_cell/ReadVariableOp_7?gru_cell/ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlicegru_cell/ReadVariableOp:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_1ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_1:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_2ReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_2?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_1?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/BiasAdd_2?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_6:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulzeros:output:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_7ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_7?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_7:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulzeros:output:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/adde
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Consti
gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_1?
gru_cell/MulMulgru_cell/add:z:0gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul?
gru_cell/Add_1AddV2gru_cell/Mul:z:0gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_1?
 gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_cell/clip_by_value/Minimum/y?
gru_cell/clip_by_value/MinimumMinimumgru_cell/Add_1:z:0)gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2 
gru_cell/clip_by_value/Minimumy
gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value/y?
gru_cell/clip_by_valueMaximum"gru_cell/clip_by_value/Minimum:z:0!gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value?
gru_cell/add_2AddV2gru_cell/BiasAdd_1:output:0gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_2i
gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell/Const_2i
gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell/Const_3?
gru_cell/Mul_1Mulgru_cell/add_2:z:0gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Mul_1?
gru_cell/Add_3AddV2gru_cell/Mul_1:z:0gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/Add_3?
"gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru_cell/clip_by_value_1/Minimum/y?
 gru_cell/clip_by_value_1/MinimumMinimumgru_cell/Add_3:z:0+gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_cell/clip_by_value_1/Minimum}
gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell/clip_by_value_1/y?
gru_cell/clip_by_value_1Maximum$gru_cell/clip_by_value_1/Minimum:z:0#gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/clip_by_value_1?
gru_cell/mul_2Mulgru_cell/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_8ReadVariableOp"gru_cell_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell/ReadVariableOp_8?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlice!gru_cell/ReadVariableOp_8:value:0'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell/strided_slice_8?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/MatMul_5?
gru_cell/add_4AddV2gru_cell/BiasAdd_2:output:0gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_4l
gru_cell/ReluRelugru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/Relu?
gru_cell/mul_3Mulgru_cell/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_3e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/sub?
gru_cell/mul_4Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell/mul_4?
gru_cell/add_5AddV2gru_cell/mul_3:z:0gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell/add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_3_resource"gru_cell_readvariableop_6_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_31377*
condR
while_cond_31376*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1n
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^gru_cell/ReadVariableOp_7^gru_cell/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_626
gru_cell/ReadVariableOp_7gru_cell/ReadVariableOp_726
gru_cell/ReadVariableOp_8gru_cell/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34173

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_33409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_33409___redundant_placeholder03
/while_while_cond_33409___redundant_placeholder13
/while_while_cond_33409___redundant_placeholder23
/while_while_cond_33409___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
2__inference_time_distributed_1_layer_call_fn_34343

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_316232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_31623

inputs8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_31564

inputs8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
(__inference_gru_cell_layer_call_fn_34549

inputs
states_0
unknown:	?H
	unknown_0:H
	unknown_1:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_305432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:??????????:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_31543

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_34121

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_30988

inputs
dense_30978: 
dense_30980: 
identity??dense/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_30978dense_30980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_309292
dense/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityn
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
I
-__inference_repeat_vector_layer_call_fn_33031

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_repeat_vector_layer_call_and_return_conditional_losses_302112
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
2__inference_time_distributed_1_layer_call_fn_34334

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_315642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
H__inference_repeat_vector_layer_call_and_return_conditional_losses_30211

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
stackx
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :??????????????????2
Tilen
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_32964

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
while_cond_33921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_33921___redundant_placeholder03
/while_while_cond_33921___redundant_placeholder13
/while_while_cond_33921___redundant_placeholder23
/while_while_cond_33921___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_31249

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_30929

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_repeat_vector_layer_call_and_return_conditional_losses_31258

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
stackp
TileTileExpandDims:output:0stack:output:0*
T0*,
_output_shapes
:??????????2
Tilef
IdentityIdentityTile:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_30609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_30609___redundant_placeholder03
/while_while_cond_30609___redundant_placeholder13
/while_while_cond_30609___redundant_placeholder23
/while_while_cond_30609___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_31127

inputs
dense_1_31117: 
dense_1_31119:
identity??dense_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_31117dense_1_31119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_310682!
dense_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityp
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?1
?
E__inference_sequential_layer_call_and_return_conditional_losses_32197
conv1d_input"
conv1d_32161: 
conv1d_32163: $
conv1d_1_32166: @
conv1d_1_32168:@
	gru_32174:	?H
	gru_32176:H
	gru_32178:H(
time_distributed_32182: $
time_distributed_32184: *
time_distributed_1_32189: &
time_distributed_1_32191:
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?gru/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_32161conv1d_32163*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_312062 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_32166conv1d_1_32168*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_312282"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_312412
max_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_312492
flatten/PartitionedCall?
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_repeat_vector_layer_call_and_return_conditional_losses_312582
repeat_vector/PartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	gru_32174	gru_32176	gru_32178*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_319542
gru/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_316822!
dropout/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0time_distributed_32182time_distributed_32184*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_316552*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape(dropout/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_32189time_distributed_1_32191*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_316232,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape?
IdentityIdentity3time_distributed_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
?
d
H__inference_repeat_vector_layer_call_and_return_conditional_losses_33026

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
stackp
TileTileExpandDims:output:0stack:output:0*
T0*,
_output_shapes
:??????????2
Tilef
IdentityIdentityTile:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34201

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_33665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_33665___redundant_placeholder03
/while_while_cond_33665___redundant_placeholder13
/while_while_cond_33665___redundant_placeholder23
/while_while_cond_33665___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?	
while_body_31377
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	?H8
*while_gru_cell_readvariableop_3_resource_0:H<
*while_gru_cell_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	?H6
(while_gru_cell_readvariableop_3_resource:H:
(while_gru_cell_readvariableop_6_resource:H??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?while/gru_cell/ReadVariableOp_7?while/gru_cell/ReadVariableOp_8?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02
while/gru_cell/ReadVariableOp?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice%while/gru_cell/ReadVariableOp:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_1ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_1?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_1:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_2ReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_2?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_1?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_6?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_6:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile_placeholder_2'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_7ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_7?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_7:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile_placeholder_2'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/addq
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Constu
while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_1?
while/gru_cell/MulMulwhile/gru_cell/add:z:0while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul?
while/gru_cell/Add_1AddV2while/gru_cell/Mul:z:0while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_1?
&while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&while/gru_cell/clip_by_value/Minimum/y?
$while/gru_cell/clip_by_value/MinimumMinimumwhile/gru_cell/Add_1:z:0/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2&
$while/gru_cell/clip_by_value/Minimum?
while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
while/gru_cell/clip_by_value/y?
while/gru_cell/clip_by_valueMaximum(while/gru_cell/clip_by_value/Minimum:z:0'while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/clip_by_value?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_2u
while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell/Const_2u
while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell/Const_3?
while/gru_cell/Mul_1Mulwhile/gru_cell/add_2:z:0while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Mul_1?
while/gru_cell/Add_3AddV2while/gru_cell/Mul_1:z:0while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Add_3?
(while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(while/gru_cell/clip_by_value_1/Minimum/y?
&while/gru_cell/clip_by_value_1/MinimumMinimumwhile/gru_cell/Add_3:z:01while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/gru_cell/clip_by_value_1/Minimum?
 while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 while/gru_cell/clip_by_value_1/y?
while/gru_cell/clip_by_value_1Maximum*while/gru_cell/clip_by_value_1/Minimum:z:0)while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2 
while/gru_cell/clip_by_value_1?
while/gru_cell/mul_2Mul"while/gru_cell/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_8ReadVariableOp*while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02!
while/gru_cell/ReadVariableOp_8?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlice'while/gru_cell/ReadVariableOp_8:value:0-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/MatMul_5?
while/gru_cell/add_4AddV2!while/gru_cell/BiasAdd_2:output:0!while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_4~
while/gru_cell/ReluReluwhile/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/Relu?
while/gru_cell/mul_3Mul while/gru_cell/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_3q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0 while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/sub?
while/gru_cell/mul_4Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/mul_4?
while/gru_cell/add_5AddV2while/gru_cell/mul_3:z:0while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6 ^while/gru_cell/ReadVariableOp_7 ^while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"V
(while_gru_cell_readvariableop_3_resource*while_gru_cell_readvariableop_3_resource_0"V
(while_gru_cell_readvariableop_6_resource*while_gru_cell_readvariableop_6_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_62B
while/gru_cell/ReadVariableOp_7while/gru_cell/ReadVariableOp_72B
while/gru_cell/ReadVariableOp_8while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_31682

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_34104

inputs
unknown:	?H
	unknown_0:H
	unknown_1:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_319542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?Z
?
__inference__traced_save_34739
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_1_kernel_read_readvariableop6
2savev2_time_distributed_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_nadam_conv1d_kernel_m_read_readvariableop2
.savev2_nadam_conv1d_bias_m_read_readvariableop6
2savev2_nadam_conv1d_1_kernel_m_read_readvariableop4
0savev2_nadam_conv1d_1_bias_m_read_readvariableop:
6savev2_nadam_gru_gru_cell_kernel_m_read_readvariableopD
@savev2_nadam_gru_gru_cell_recurrent_kernel_m_read_readvariableop8
4savev2_nadam_gru_gru_cell_bias_m_read_readvariableop>
:savev2_nadam_time_distributed_kernel_m_read_readvariableop<
8savev2_nadam_time_distributed_bias_m_read_readvariableop@
<savev2_nadam_time_distributed_1_kernel_m_read_readvariableop>
:savev2_nadam_time_distributed_1_bias_m_read_readvariableop4
0savev2_nadam_conv1d_kernel_v_read_readvariableop2
.savev2_nadam_conv1d_bias_v_read_readvariableop6
2savev2_nadam_conv1d_1_kernel_v_read_readvariableop4
0savev2_nadam_conv1d_1_bias_v_read_readvariableop:
6savev2_nadam_gru_gru_cell_kernel_v_read_readvariableopD
@savev2_nadam_gru_gru_cell_recurrent_kernel_v_read_readvariableop8
4savev2_nadam_gru_gru_cell_bias_v_read_readvariableop>
:savev2_nadam_time_distributed_kernel_v_read_readvariableop<
8savev2_nadam_time_distributed_bias_v_read_readvariableop@
<savev2_nadam_time_distributed_1_kernel_v_read_readvariableop>
:savev2_nadam_time_distributed_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_1_kernel_read_readvariableop2savev2_time_distributed_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_nadam_conv1d_kernel_m_read_readvariableop.savev2_nadam_conv1d_bias_m_read_readvariableop2savev2_nadam_conv1d_1_kernel_m_read_readvariableop0savev2_nadam_conv1d_1_bias_m_read_readvariableop6savev2_nadam_gru_gru_cell_kernel_m_read_readvariableop@savev2_nadam_gru_gru_cell_recurrent_kernel_m_read_readvariableop4savev2_nadam_gru_gru_cell_bias_m_read_readvariableop:savev2_nadam_time_distributed_kernel_m_read_readvariableop8savev2_nadam_time_distributed_bias_m_read_readvariableop<savev2_nadam_time_distributed_1_kernel_m_read_readvariableop:savev2_nadam_time_distributed_1_bias_m_read_readvariableop0savev2_nadam_conv1d_kernel_v_read_readvariableop.savev2_nadam_conv1d_bias_v_read_readvariableop2savev2_nadam_conv1d_1_kernel_v_read_readvariableop0savev2_nadam_conv1d_1_bias_v_read_readvariableop6savev2_nadam_gru_gru_cell_kernel_v_read_readvariableop@savev2_nadam_gru_gru_cell_recurrent_kernel_v_read_readvariableop4savev2_nadam_gru_gru_cell_bias_v_read_readvariableop:savev2_nadam_time_distributed_kernel_v_read_readvariableop8savev2_nadam_time_distributed_bias_v_read_readvariableop<savev2_nadam_time_distributed_1_kernel_v_read_readvariableop:savev2_nadam_time_distributed_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@: : : : : : :	?H:H:H: : : :: : : : : : : @:@:	?H:H:H: : : :: : : @:@:	?H:H:H: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	?H:$ 

_output_shapes

:H: 

_output_shapes
:H:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?H:$ 

_output_shapes

:H: 

_output_shapes
:H:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: :  

_output_shapes
::(!$
"
_output_shapes
: : "

_output_shapes
: :(#$
"
_output_shapes
: @: $

_output_shapes
:@:%%!

_output_shapes
:	?H:$& 

_output_shapes

:H: '

_output_shapes
:H:$( 

_output_shapes

: : )

_output_shapes
: :$* 

_output_shapes

: : +

_output_shapes
::,

_output_shapes
: 
?a
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_30543

inputs

states*
readvariableop_resource:	?H'
readvariableop_3_resource:H+
readvariableop_6_resource:H
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	?H*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddz
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1z
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6r
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3~
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7r
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Muld
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1f
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1d
mul_2Mulclip_by_value_1:z:0states*
T0*'
_output_shapes
:?????????2
mul_2~
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes

:H*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:?????????2
Relub
mul_3Mulclip_by_value:z:0states*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subd
mul_4Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5d
IdentityIdentity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh

Identity_1Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:??????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
?
0__inference_time_distributed_layer_call_fn_34219

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_309882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_34587

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_310682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
gru_while_cond_32698$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_32698___redundant_placeholder0;
7gru_while_gru_while_cond_32698___redundant_placeholder1;
7gru_while_gru_while_cond_32698___redundant_placeholder2;
7gru_while_gru_while_cond_32698___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_32939

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
E__inference_sequential_layer_call_and_return_conditional_losses_31573

inputs"
conv1d_31207: 
conv1d_31209: $
conv1d_1_31229: @
conv1d_1_31231:@
	gru_31516:	?H
	gru_31518:H
	gru_31520:H(
time_distributed_31544: $
time_distributed_31546: *
time_distributed_1_31565: &
time_distributed_1_31567:
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?gru/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_31207conv1d_31209*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_312062 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_31229conv1d_1_31231*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_312282"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_312412
max_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_312492
flatten/PartitionedCall?
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_repeat_vector_layer_call_and_return_conditional_losses_312582
repeat_vector/PartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	gru_31516	gru_31518	gru_31520*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_315152
gru/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_315282
dropout/PartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0time_distributed_31544time_distributed_31546*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_315432*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape dropout/PartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_31565time_distributed_1_31567*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_315642,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_1/Reshape?
IdentityIdentity3time_distributed_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_32232
conv1d_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	?H
	unknown_4:H
	unknown_5:H
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_301712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_32981

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ɧ
?

gru_while_body_32699$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	?H<
.gru_while_gru_cell_readvariableop_3_resource_0:H@
.gru_while_gru_cell_readvariableop_6_resource_0:H
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	?H:
,gru_while_gru_cell_readvariableop_3_resource:H>
,gru_while_gru_cell_readvariableop_6_resource:H??!gru/while/gru_cell/ReadVariableOp?#gru/while/gru_cell/ReadVariableOp_1?#gru/while/gru_cell/ReadVariableOp_2?#gru/while/gru_cell/ReadVariableOp_3?#gru/while/gru_cell/ReadVariableOp_4?#gru/while/gru_cell/ReadVariableOp_5?#gru/while/gru_cell/ReadVariableOp_6?#gru/while/gru_cell/ReadVariableOp_7?#gru/while/gru_cell/ReadVariableOp_8?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack?
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(gru/while/gru_cell/strided_slice/stack_1?
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2?
 gru/while/gru_cell/strided_sliceStridedSlice)gru/while/gru_cell/ReadVariableOp:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul?
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1?
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru/while/gru_cell/strided_slice_1/stack?
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru/while/gru_cell/strided_slice_1/stack_1?
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2?
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1?
gru/while/gru_cell/MatMul_1MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_1?
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2?
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru/while/gru_cell/strided_slice_2/stack?
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1?
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2?
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2?
gru/while/gru_cell/MatMul_2MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_2?
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3?
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack?
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_1?
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2?
"gru/while/gru_cell/strided_slice_3StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"gru/while/gru_cell/strided_slice_3?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/BiasAdd?
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4?
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(gru/while/gru_cell/strided_slice_4/stack?
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02,
*gru/while/gru_cell/strided_slice_4/stack_1?
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2?
"gru/while/gru_cell/strided_slice_4StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/strided_slice_4?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/BiasAdd_1?
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5?
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02*
(gru/while/gru_cell/strided_slice_5/stack?
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1?
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2?
"gru/while/gru_cell/strided_slice_5StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2$
"gru/while/gru_cell/strided_slice_5?
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/BiasAdd_2?
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6?
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack?
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*gru/while/gru_cell/strided_slice_6/stack_1?
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2?
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6?
gru/while/gru_cell/MatMul_3MatMulgru_while_placeholder_2+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_3?
#gru/while/gru_cell/ReadVariableOp_7ReadVariableOp.gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_7?
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru/while/gru_cell/strided_slice_7/stack?
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru/while/gru_cell/strided_slice_7/stack_1?
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2?
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_7:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7?
gru/while/gru_cell/MatMul_4MatMulgru_while_placeholder_2+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_4?
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/addy
gru/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/while/gru_cell/Const}
gru/while/gru_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/while/gru_cell/Const_1?
gru/while/gru_cell/MulMulgru/while/gru_cell/add:z:0!gru/while/gru_cell/Const:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Mul?
gru/while/gru_cell/Add_1AddV2gru/while/gru_cell/Mul:z:0#gru/while/gru_cell/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Add_1?
*gru/while/gru_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*gru/while/gru_cell/clip_by_value/Minimum/y?
(gru/while/gru_cell/clip_by_value/MinimumMinimumgru/while/gru_cell/Add_1:z:03gru/while/gru_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(gru/while/gru_cell/clip_by_value/Minimum?
"gru/while/gru_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"gru/while/gru_cell/clip_by_value/y?
 gru/while/gru_cell/clip_by_valueMaximum,gru/while/gru_cell/clip_by_value/Minimum:z:0+gru/while/gru_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru/while/gru_cell/clip_by_value?
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/add_2}
gru/while/gru_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/while/gru_cell/Const_2}
gru/while/gru_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/while/gru_cell/Const_3?
gru/while/gru_cell/Mul_1Mulgru/while/gru_cell/add_2:z:0#gru/while/gru_cell/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Mul_1?
gru/while/gru_cell/Add_3AddV2gru/while/gru_cell/Mul_1:z:0#gru/while/gru_cell/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Add_3?
,gru/while/gru_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,gru/while/gru_cell/clip_by_value_1/Minimum/y?
*gru/while/gru_cell/clip_by_value_1/MinimumMinimumgru/while/gru_cell/Add_3:z:05gru/while/gru_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2,
*gru/while/gru_cell/clip_by_value_1/Minimum?
$gru/while/gru_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$gru/while/gru_cell/clip_by_value_1/y?
"gru/while/gru_cell/clip_by_value_1Maximum.gru/while/gru_cell/clip_by_value_1/Minimum:z:0-gru/while/gru_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2$
"gru/while/gru_cell/clip_by_value_1?
gru/while/gru_cell/mul_2Mul&gru/while/gru_cell/clip_by_value_1:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/mul_2?
#gru/while/gru_cell/ReadVariableOp_8ReadVariableOp.gru_while_gru_cell_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02%
#gru/while/gru_cell/ReadVariableOp_8?
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru/while/gru_cell/strided_slice_8/stack?
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_8/stack_1?
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_8/stack_2?
"gru/while/gru_cell/strided_slice_8StridedSlice+gru/while/gru_cell/ReadVariableOp_8:value:01gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_8?
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/MatMul_5?
gru/while/gru_cell/add_4AddV2%gru/while/gru_cell/BiasAdd_2:output:0%gru/while/gru_cell/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/add_4?
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/Relu?
gru/while/gru_cell/mul_3Mul$gru/while/gru_cell/clip_by_value:z:0gru_while_placeholder_2*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/mul_3y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0$gru/while/gru_cell/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_4Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/mul_4?
gru/while/gru_cell/add_5AddV2gru/while/gru_cell/mul_3:z:0gru/while/gru_cell/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru/while/gru_cell/add_5?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_5:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1{
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity_1}
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_5:z:0^gru/while/NoOp*
T0*'
_output_shapes
:?????????2
gru/while/Identity_4?
gru/while/NoOpNoOp"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6$^gru/while/gru_cell/ReadVariableOp_7$^gru/while/gru_cell/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
gru/while/NoOp"^
,gru_while_gru_cell_readvariableop_3_resource.gru_while_gru_cell_readvariableop_3_resource_0"^
,gru_while_gru_cell_readvariableop_6_resource.gru_while_gru_cell_readvariableop_6_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_62J
#gru/while/gru_cell/ReadVariableOp_7#gru/while/gru_cell/ReadVariableOp_72J
#gru/while/gru_cell/ReadVariableOp_8#gru/while/gru_cell/ReadVariableOp_8: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_sequential_layer_call_fn_32896

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	?H
	unknown_4:H
	unknown_5:H
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_315732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_30940

inputs
dense_30930: 
dense_30932: 
identity??dense/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_30930dense_30932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_309292
dense/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityn
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_33010

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_312492
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_30183

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_31598
conv1d_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	?H
	unknown_4:H
	unknown_5:H
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_315732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
?
?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34187

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_34109

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34293

inputs8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?=
?
>__inference_gru_layer_call_and_return_conditional_losses_30427

inputs!
gru_cell_30352:	?H
gru_cell_30354:H 
gru_cell_30356:H
identity?? gru_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_30352gru_cell_30354gru_cell_30356*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_303512"
 gru_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_30352gru_cell_30354gru_cell_30356*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_30364*
condR
while_cond_30363*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1w
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityy
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
conv1d_input9
serving_default_conv1d_input:0?????????J
time_distributed_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
(cell
)
state_spec
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
.regularization_losses
/	variables
0trainable_variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	2layer
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	7layer
8regularization_losses
9	variables
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
<iter

=beta_1

>beta_2
	?decay
@learning_rate
Amomentum_cachem?m?m?m?Bm?Cm?Dm?Em?Fm?Gm?Hm?v?v?v?v?Bv?Cv?Dv?Ev?Fv?Gv?Hv?"
	optimizer
 "
trackable_list_wrapper
n
0
1
2
3
B4
C5
D6
E7
F8
G9
H10"
trackable_list_wrapper
n
0
1
2
3
B4
C5
D6
E7
F8
G9
H10"
trackable_list_wrapper
?
regularization_losses
Ilayer_metrics
	variables
trainable_variables
Jnon_trainable_variables
Klayer_regularization_losses
Lmetrics

Mlayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:! 2conv1d/kernel
: 2conv1d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Nlayer_metrics
	variables
trainable_variables
Onon_trainable_variables
Player_regularization_losses
Qmetrics

Rlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:# @2conv1d_1/kernel
:@2conv1d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Slayer_metrics
	variables
trainable_variables
Tnon_trainable_variables
Ulayer_regularization_losses
Vmetrics

Wlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Xlayer_metrics
	variables
trainable_variables
Ynon_trainable_variables
Zlayer_regularization_losses
[metrics

\layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 regularization_losses
]layer_metrics
!	variables
"trainable_variables
^non_trainable_variables
_layer_regularization_losses
`metrics

alayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$regularization_losses
blayer_metrics
%	variables
&trainable_variables
cnon_trainable_variables
dlayer_regularization_losses
emetrics

flayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Bkernel
Crecurrent_kernel
Dbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
B0
C1
D2"
trackable_list_wrapper
5
B0
C1
D2"
trackable_list_wrapper
?
*regularization_losses
klayer_metrics
+	variables
,trainable_variables
lnon_trainable_variables
mlayer_regularization_losses

nstates
ometrics

players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
.regularization_losses
qlayer_metrics
/	variables
0trainable_variables
rnon_trainable_variables
slayer_regularization_losses
tmetrics

ulayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Ekernel
Fbias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
3regularization_losses
zlayer_metrics
4	variables
5trainable_variables
{non_trainable_variables
|layer_regularization_losses
}metrics

~layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Gkernel
Hbias
regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
8regularization_losses
?layer_metrics
9	variables
:trainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
&:$	?H2gru/gru_cell/kernel
/:-H2gru/gru_cell/recurrent_kernel
:H2gru/gru_cell/bias
):' 2time_distributed/kernel
#:! 2time_distributed/bias
+:) 2time_distributed_1/kernel
%:#2time_distributed_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
B0
C1
D2"
trackable_list_wrapper
5
B0
C1
D2"
trackable_list_wrapper
?
gregularization_losses
?layer_metrics
h	variables
itrainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
vregularization_losses
?layer_metrics
w	variables
xtrainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
regularization_losses
?layer_metrics
?	variables
?trainable_variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
70"
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
):' 2Nadam/conv1d/kernel/m
: 2Nadam/conv1d/bias/m
+:) @2Nadam/conv1d_1/kernel/m
!:@2Nadam/conv1d_1/bias/m
,:*	?H2Nadam/gru/gru_cell/kernel/m
5:3H2%Nadam/gru/gru_cell/recurrent_kernel/m
%:#H2Nadam/gru/gru_cell/bias/m
/:- 2Nadam/time_distributed/kernel/m
):' 2Nadam/time_distributed/bias/m
1:/ 2!Nadam/time_distributed_1/kernel/m
+:)2Nadam/time_distributed_1/bias/m
):' 2Nadam/conv1d/kernel/v
: 2Nadam/conv1d/bias/v
+:) @2Nadam/conv1d_1/kernel/v
!:@2Nadam/conv1d_1/bias/v
,:*	?H2Nadam/gru/gru_cell/kernel/v
5:3H2%Nadam/gru/gru_cell/recurrent_kernel/v
%:#H2Nadam/gru/gru_cell/bias/v
/:- 2Nadam/time_distributed/kernel/v
):' 2Nadam/time_distributed/bias/v
1:/ 2!Nadam/time_distributed_1/kernel/v
+:)2Nadam/time_distributed_1/bias/v
?B?
 __inference__wrapped_model_30171conv1d_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_32547
E__inference_sequential_layer_call_and_return_conditional_losses_32869
E__inference_sequential_layer_call_and_return_conditional_losses_32158
E__inference_sequential_layer_call_and_return_conditional_losses_32197?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_sequential_layer_call_fn_31598
*__inference_sequential_layer_call_fn_32896
*__inference_sequential_layer_call_fn_32923
*__inference_sequential_layer_call_fn_32119?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_conv1d_layer_call_and_return_conditional_losses_32939?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv1d_layer_call_fn_32948?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_32964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv1d_1_layer_call_fn_32973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_32981
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_32989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_max_pooling1d_layer_call_fn_32994
-__inference_max_pooling1d_layer_call_fn_32999?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_33005?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_33010?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_repeat_vector_layer_call_and_return_conditional_losses_33018
H__inference_repeat_vector_layer_call_and_return_conditional_losses_33026?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_repeat_vector_layer_call_fn_33031
-__inference_repeat_vector_layer_call_fn_33036?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_gru_layer_call_and_return_conditional_losses_33292
>__inference_gru_layer_call_and_return_conditional_losses_33548
>__inference_gru_layer_call_and_return_conditional_losses_33804
>__inference_gru_layer_call_and_return_conditional_losses_34060?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_gru_layer_call_fn_34071
#__inference_gru_layer_call_fn_34082
#__inference_gru_layer_call_fn_34093
#__inference_gru_layer_call_fn_34104?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_34109
B__inference_dropout_layer_call_and_return_conditional_losses_34121?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_34126
'__inference_dropout_layer_call_fn_34131?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34152
K__inference_time_distributed_layer_call_and_return_conditional_losses_34173
K__inference_time_distributed_layer_call_and_return_conditional_losses_34187
K__inference_time_distributed_layer_call_and_return_conditional_losses_34201?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_time_distributed_layer_call_fn_34210
0__inference_time_distributed_layer_call_fn_34219
0__inference_time_distributed_layer_call_fn_34228
0__inference_time_distributed_layer_call_fn_34237?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34258
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34279
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34293
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34307?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_time_distributed_1_layer_call_fn_34316
2__inference_time_distributed_1_layer_call_fn_34325
2__inference_time_distributed_1_layer_call_fn_34334
2__inference_time_distributed_1_layer_call_fn_34343?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_32232conv1d_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_gru_cell_layer_call_and_return_conditional_losses_34432
C__inference_gru_cell_layer_call_and_return_conditional_losses_34521?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_gru_cell_layer_call_fn_34535
(__inference_gru_cell_layer_call_fn_34549?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_34559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_34568?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_34578?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_34587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_30171?BDCEFGH9?6
/?,
*?'
conv1d_input?????????
? "K?H
F
time_distributed_10?-
time_distributed_1??????????
C__inference_conv1d_1_layer_call_and_return_conditional_losses_32964d3?0
)?&
$?!
inputs????????? 
? ")?&
?
0?????????@
? ?
(__inference_conv1d_1_layer_call_fn_32973W3?0
)?&
$?!
inputs????????? 
? "??????????@?
A__inference_conv1d_layer_call_and_return_conditional_losses_32939d3?0
)?&
$?!
inputs?????????
? ")?&
?
0????????? 
? ?
&__inference_conv1d_layer_call_fn_32948W3?0
)?&
$?!
inputs?????????
? "?????????? ?
B__inference_dense_1_layer_call_and_return_conditional_losses_34578\GH/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_34587OGH/?,
%?"
 ?
inputs????????? 
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_34559\EF/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? x
%__inference_dense_layer_call_fn_34568OEF/?,
%?"
 ?
inputs?????????
? "?????????? ?
B__inference_dropout_layer_call_and_return_conditional_losses_34109d7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_34121d7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
'__inference_dropout_layer_call_fn_34126W7?4
-?*
$?!
inputs?????????
p 
? "???????????
'__inference_dropout_layer_call_fn_34131W7?4
-?*
$?!
inputs?????????
p
? "???????????
B__inference_flatten_layer_call_and_return_conditional_losses_33005]3?0
)?&
$?!
inputs?????????@
? "&?#
?
0??????????
? {
'__inference_flatten_layer_call_fn_33010P3?0
)?&
$?!
inputs?????????@
? "????????????
C__inference_gru_cell_layer_call_and_return_conditional_losses_34432?BDC]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????
p 
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
C__inference_gru_cell_layer_call_and_return_conditional_losses_34521?BDC]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????
p
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
(__inference_gru_cell_layer_call_fn_34535?BDC]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????
p 
? "D?A
?
0?????????
"?
?
1/0??????????
(__inference_gru_cell_layer_call_fn_34549?BDC]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????
p
? "D?A
?
0?????????
"?
?
1/0??????????
>__inference_gru_layer_call_and_return_conditional_losses_33292?BDCP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "2?/
(?%
0??????????????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_33548?BDCP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "2?/
(?%
0??????????????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_33804rBDC@?=
6?3
%?"
inputs??????????

 
p 

 
? ")?&
?
0?????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_34060rBDC@?=
6?3
%?"
inputs??????????

 
p

 
? ")?&
?
0?????????
? ?
#__inference_gru_layer_call_fn_34071~BDCP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "%?"???????????????????
#__inference_gru_layer_call_fn_34082~BDCP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "%?"???????????????????
#__inference_gru_layer_call_fn_34093eBDC@?=
6?3
%?"
inputs??????????

 
p 

 
? "???????????
#__inference_gru_layer_call_fn_34104eBDC@?=
6?3
%?"
inputs??????????

 
p

 
? "???????????
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_32981?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_32989`3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
-__inference_max_pooling1d_layer_call_fn_32994wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
-__inference_max_pooling1d_layer_call_fn_32999S3?0
)?&
$?!
inputs?????????@
? "??????????@?
H__inference_repeat_vector_layer_call_and_return_conditional_losses_33018n8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
H__inference_repeat_vector_layer_call_and_return_conditional_losses_33026^0?-
&?#
!?
inputs??????????
? "*?'
 ?
0??????????
? ?
-__inference_repeat_vector_layer_call_fn_33031a8?5
.?+
)?&
inputs??????????????????
? "%?"???????????????????
-__inference_repeat_vector_layer_call_fn_33036Q0?-
&?#
!?
inputs??????????
? "????????????
E__inference_sequential_layer_call_and_return_conditional_losses_32158{BDCEFGHA?>
7?4
*?'
conv1d_input?????????
p 

 
? ")?&
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_32197{BDCEFGHA?>
7?4
*?'
conv1d_input?????????
p

 
? ")?&
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_32547uBDCEFGH;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_32869uBDCEFGH;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
*__inference_sequential_layer_call_fn_31598nBDCEFGHA?>
7?4
*?'
conv1d_input?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_32119nBDCEFGHA?>
7?4
*?'
conv1d_input?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_32896hBDCEFGH;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_32923hBDCEFGH;?8
1?.
$?!
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_32232?BDCEFGHI?F
? 
??<
:
conv1d_input*?'
conv1d_input?????????"K?H
F
time_distributed_10?-
time_distributed_1??????????
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34258~GHD?A
:?7
-?*
inputs?????????????????? 
p 

 
? "2?/
(?%
0??????????????????
? ?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34279~GHD?A
:?7
-?*
inputs?????????????????? 
p

 
? "2?/
(?%
0??????????????????
? ?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34293lGH;?8
1?.
$?!
inputs????????? 
p 

 
? ")?&
?
0?????????
? ?
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34307lGH;?8
1?.
$?!
inputs????????? 
p

 
? ")?&
?
0?????????
? ?
2__inference_time_distributed_1_layer_call_fn_34316qGHD?A
:?7
-?*
inputs?????????????????? 
p 

 
? "%?"???????????????????
2__inference_time_distributed_1_layer_call_fn_34325qGHD?A
:?7
-?*
inputs?????????????????? 
p

 
? "%?"???????????????????
2__inference_time_distributed_1_layer_call_fn_34334_GH;?8
1?.
$?!
inputs????????? 
p 

 
? "???????????
2__inference_time_distributed_1_layer_call_fn_34343_GH;?8
1?.
$?!
inputs????????? 
p

 
? "???????????
K__inference_time_distributed_layer_call_and_return_conditional_losses_34152~EFD?A
:?7
-?*
inputs??????????????????
p 

 
? "2?/
(?%
0?????????????????? 
? ?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34173~EFD?A
:?7
-?*
inputs??????????????????
p

 
? "2?/
(?%
0?????????????????? 
? ?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34187lEF;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0????????? 
? ?
K__inference_time_distributed_layer_call_and_return_conditional_losses_34201lEF;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0????????? 
? ?
0__inference_time_distributed_layer_call_fn_34210qEFD?A
:?7
-?*
inputs??????????????????
p 

 
? "%?"?????????????????? ?
0__inference_time_distributed_layer_call_fn_34219qEFD?A
:?7
-?*
inputs??????????????????
p

 
? "%?"?????????????????? ?
0__inference_time_distributed_layer_call_fn_34228_EF;?8
1?.
$?!
inputs?????????
p 

 
? "?????????? ?
0__inference_time_distributed_layer_call_fn_34237_EF;?8
1?.
$?!
inputs?????????
p

 
? "?????????? 