??.
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
?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??,
?
conv1d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv1d_24/kernel
y
$conv1d_24/kernel/Read/ReadVariableOpReadVariableOpconv1d_24/kernel*"
_output_shapes
: *
dtype0
t
conv1d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_24/bias
m
"conv1d_24/bias/Read/ReadVariableOpReadVariableOpconv1d_24/bias*
_output_shapes
: *
dtype0
?
conv1d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv1d_25/kernel
y
$conv1d_25/kernel/Read/ReadVariableOpReadVariableOpconv1d_25/kernel*"
_output_shapes
: @*
dtype0
t
conv1d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_25/bias
m
"conv1d_25/bias/Read/ReadVariableOpReadVariableOpconv1d_25/bias*
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
gru_12/gru_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H**
shared_namegru_12/gru_cell_12/kernel
?
-gru_12/gru_cell_12/kernel/Read/ReadVariableOpReadVariableOpgru_12/gru_cell_12/kernel*
_output_shapes
:	?H*
dtype0
?
#gru_12/gru_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*4
shared_name%#gru_12/gru_cell_12/recurrent_kernel
?
7gru_12/gru_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_12/gru_cell_12/recurrent_kernel*
_output_shapes

:H*
dtype0
?
gru_12/gru_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*(
shared_namegru_12/gru_cell_12/bias

+gru_12/gru_cell_12/bias/Read/ReadVariableOpReadVariableOpgru_12/gru_cell_12/bias*
_output_shapes
:H*
dtype0
?
time_distributed_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nametime_distributed_24/kernel
?
.time_distributed_24/kernel/Read/ReadVariableOpReadVariableOptime_distributed_24/kernel*
_output_shapes

: *
dtype0
?
time_distributed_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametime_distributed_24/bias
?
,time_distributed_24/bias/Read/ReadVariableOpReadVariableOptime_distributed_24/bias*
_output_shapes
: *
dtype0
?
time_distributed_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nametime_distributed_25/kernel
?
.time_distributed_25/kernel/Read/ReadVariableOpReadVariableOptime_distributed_25/kernel*
_output_shapes

: *
dtype0
?
time_distributed_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nametime_distributed_25/bias
?
,time_distributed_25/bias/Read/ReadVariableOpReadVariableOptime_distributed_25/bias*
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
Nadam/conv1d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameNadam/conv1d_24/kernel/m
?
,Nadam/conv1d_24/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_24/kernel/m*"
_output_shapes
: *
dtype0
?
Nadam/conv1d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameNadam/conv1d_24/bias/m
}
*Nadam/conv1d_24/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_24/bias/m*
_output_shapes
: *
dtype0
?
Nadam/conv1d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameNadam/conv1d_25/kernel/m
?
,Nadam/conv1d_25/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_25/kernel/m*"
_output_shapes
: @*
dtype0
?
Nadam/conv1d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameNadam/conv1d_25/bias/m
}
*Nadam/conv1d_25/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_25/bias/m*
_output_shapes
:@*
dtype0
?
!Nadam/gru_12/gru_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*2
shared_name#!Nadam/gru_12/gru_cell_12/kernel/m
?
5Nadam/gru_12/gru_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/gru_12/gru_cell_12/kernel/m*
_output_shapes
:	?H*
dtype0
?
+Nadam/gru_12/gru_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*<
shared_name-+Nadam/gru_12/gru_cell_12/recurrent_kernel/m
?
?Nadam/gru_12/gru_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp+Nadam/gru_12/gru_cell_12/recurrent_kernel/m*
_output_shapes

:H*
dtype0
?
Nadam/gru_12/gru_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*0
shared_name!Nadam/gru_12/gru_cell_12/bias/m
?
3Nadam/gru_12/gru_cell_12/bias/m/Read/ReadVariableOpReadVariableOpNadam/gru_12/gru_cell_12/bias/m*
_output_shapes
:H*
dtype0
?
"Nadam/time_distributed_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *3
shared_name$"Nadam/time_distributed_24/kernel/m
?
6Nadam/time_distributed_24/kernel/m/Read/ReadVariableOpReadVariableOp"Nadam/time_distributed_24/kernel/m*
_output_shapes

: *
dtype0
?
 Nadam/time_distributed_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Nadam/time_distributed_24/bias/m
?
4Nadam/time_distributed_24/bias/m/Read/ReadVariableOpReadVariableOp Nadam/time_distributed_24/bias/m*
_output_shapes
: *
dtype0
?
"Nadam/time_distributed_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *3
shared_name$"Nadam/time_distributed_25/kernel/m
?
6Nadam/time_distributed_25/kernel/m/Read/ReadVariableOpReadVariableOp"Nadam/time_distributed_25/kernel/m*
_output_shapes

: *
dtype0
?
 Nadam/time_distributed_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Nadam/time_distributed_25/bias/m
?
4Nadam/time_distributed_25/bias/m/Read/ReadVariableOpReadVariableOp Nadam/time_distributed_25/bias/m*
_output_shapes
:*
dtype0
?
Nadam/conv1d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameNadam/conv1d_24/kernel/v
?
,Nadam/conv1d_24/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_24/kernel/v*"
_output_shapes
: *
dtype0
?
Nadam/conv1d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameNadam/conv1d_24/bias/v
}
*Nadam/conv1d_24/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_24/bias/v*
_output_shapes
: *
dtype0
?
Nadam/conv1d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameNadam/conv1d_25/kernel/v
?
,Nadam/conv1d_25/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_25/kernel/v*"
_output_shapes
: @*
dtype0
?
Nadam/conv1d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameNadam/conv1d_25/bias/v
}
*Nadam/conv1d_25/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_25/bias/v*
_output_shapes
:@*
dtype0
?
!Nadam/gru_12/gru_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*2
shared_name#!Nadam/gru_12/gru_cell_12/kernel/v
?
5Nadam/gru_12/gru_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/gru_12/gru_cell_12/kernel/v*
_output_shapes
:	?H*
dtype0
?
+Nadam/gru_12/gru_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*<
shared_name-+Nadam/gru_12/gru_cell_12/recurrent_kernel/v
?
?Nadam/gru_12/gru_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp+Nadam/gru_12/gru_cell_12/recurrent_kernel/v*
_output_shapes

:H*
dtype0
?
Nadam/gru_12/gru_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*0
shared_name!Nadam/gru_12/gru_cell_12/bias/v
?
3Nadam/gru_12/gru_cell_12/bias/v/Read/ReadVariableOpReadVariableOpNadam/gru_12/gru_cell_12/bias/v*
_output_shapes
:H*
dtype0
?
"Nadam/time_distributed_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *3
shared_name$"Nadam/time_distributed_24/kernel/v
?
6Nadam/time_distributed_24/kernel/v/Read/ReadVariableOpReadVariableOp"Nadam/time_distributed_24/kernel/v*
_output_shapes

: *
dtype0
?
 Nadam/time_distributed_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Nadam/time_distributed_24/bias/v
?
4Nadam/time_distributed_24/bias/v/Read/ReadVariableOpReadVariableOp Nadam/time_distributed_24/bias/v*
_output_shapes
: *
dtype0
?
"Nadam/time_distributed_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *3
shared_name$"Nadam/time_distributed_25/kernel/v
?
6Nadam/time_distributed_25/kernel/v/Read/ReadVariableOpReadVariableOp"Nadam/time_distributed_25/kernel/v*
_output_shapes

: *
dtype0
?
 Nadam/time_distributed_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Nadam/time_distributed_25/bias/v
?
4Nadam/time_distributed_25/bias/v/Read/ReadVariableOpReadVariableOp Nadam/time_distributed_25/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?N
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?M
value?MB?M B?M
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
l
(cell
)
state_spec
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
]
	2layer
3	variables
4trainable_variables
5regularization_losses
6	keras_api
]
	7layer
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?
<iter

=beta_1

>beta_2
	?decay
@learning_rate
Amomentum_cachem?m?m?m?Bm?Cm?Dm?Em?Fm?Gm?Hm?v?v?v?v?Bv?Cv?Dv?Ev?Fv?Gv?Hv?
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
?

Ilayers
trainable_variables
regularization_losses
Jlayer_regularization_losses
Kmetrics
	variables
Lnon_trainable_variables
Mlayer_metrics
 
\Z
VARIABLE_VALUEconv1d_24/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_24/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables

Nlayers
trainable_variables
regularization_losses
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
Rnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_25/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_25/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables

Slayers
trainable_variables
regularization_losses
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
Wnon_trainable_variables
 
 
 
?
	variables

Xlayers
trainable_variables
regularization_losses
Ylayer_regularization_losses
Zmetrics
[layer_metrics
\non_trainable_variables
 
 
 
?
 	variables

]layers
!trainable_variables
"regularization_losses
^layer_regularization_losses
_metrics
`layer_metrics
anon_trainable_variables
 
 
 
?
$	variables

blayers
%trainable_variables
&regularization_losses
clayer_regularization_losses
dmetrics
elayer_metrics
fnon_trainable_variables
~

Bkernel
Crecurrent_kernel
Dbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
 

B0
C1
D2
 

B0
C1
D2
?
*trainable_variables

klayers

lstates
+regularization_losses
mlayer_regularization_losses
nmetrics
,	variables
onon_trainable_variables
player_metrics
 
 
 
?
.	variables

qlayers
/trainable_variables
0regularization_losses
rlayer_regularization_losses
smetrics
tlayer_metrics
unon_trainable_variables
h

Ekernel
Fbias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api

E0
F1

E0
F1
 
?
3	variables

zlayers
4trainable_variables
5regularization_losses
{layer_regularization_losses
|metrics
}layer_metrics
~non_trainable_variables
k

Gkernel
Hbias
	variables
?trainable_variables
?regularization_losses
?	keras_api

G0
H1

G0
H1
 
?
8	variables
?layers
9trainable_variables
:regularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
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
_]
VARIABLE_VALUEgru_12/gru_cell_12/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_12/gru_cell_12/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_12/gru_cell_12/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEtime_distributed_24/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEtime_distributed_24/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEtime_distributed_25/kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEtime_distributed_25/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
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

?0
?1
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
 

B0
C1
D2

B0
C1
D2
 
?
g	variables
?layers
htrainable_variables
iregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables

(0
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

E0
F1

E0
F1
 
?
v	variables
?layers
wtrainable_variables
xregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables

20
 
 
 
 

G0
H1

G0
H1
 
?
	variables
?layers
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables

70
 
 
 
 
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
?~
VARIABLE_VALUENadam/conv1d_24/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_24/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/conv1d_25/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_25/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Nadam/gru_12/gru_cell_12/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Nadam/gru_12/gru_cell_12/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/gru_12/gru_cell_12/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/time_distributed_24/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Nadam/time_distributed_24/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/time_distributed_25/kernel/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Nadam/time_distributed_25/bias/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/conv1d_24/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_24/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/conv1d_25/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_25/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Nadam/gru_12/gru_cell_12/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Nadam/gru_12/gru_cell_12/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/gru_12/gru_cell_12/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/time_distributed_24/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Nadam/time_distributed_24/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/time_distributed_25/kernel/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Nadam/time_distributed_25/bias/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv1d_24_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_24_inputconv1d_24/kernelconv1d_24/biasconv1d_25/kernelconv1d_25/biasgru_12/gru_cell_12/kernelgru_12/gru_cell_12/bias#gru_12/gru_cell_12/recurrent_kerneltime_distributed_24/kerneltime_distributed_24/biastime_distributed_25/kerneltime_distributed_25/bias*
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
GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_290790
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_24/kernel/Read/ReadVariableOp"conv1d_24/bias/Read/ReadVariableOp$conv1d_25/kernel/Read/ReadVariableOp"conv1d_25/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOp-gru_12/gru_cell_12/kernel/Read/ReadVariableOp7gru_12/gru_cell_12/recurrent_kernel/Read/ReadVariableOp+gru_12/gru_cell_12/bias/Read/ReadVariableOp.time_distributed_24/kernel/Read/ReadVariableOp,time_distributed_24/bias/Read/ReadVariableOp.time_distributed_25/kernel/Read/ReadVariableOp,time_distributed_25/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Nadam/conv1d_24/kernel/m/Read/ReadVariableOp*Nadam/conv1d_24/bias/m/Read/ReadVariableOp,Nadam/conv1d_25/kernel/m/Read/ReadVariableOp*Nadam/conv1d_25/bias/m/Read/ReadVariableOp5Nadam/gru_12/gru_cell_12/kernel/m/Read/ReadVariableOp?Nadam/gru_12/gru_cell_12/recurrent_kernel/m/Read/ReadVariableOp3Nadam/gru_12/gru_cell_12/bias/m/Read/ReadVariableOp6Nadam/time_distributed_24/kernel/m/Read/ReadVariableOp4Nadam/time_distributed_24/bias/m/Read/ReadVariableOp6Nadam/time_distributed_25/kernel/m/Read/ReadVariableOp4Nadam/time_distributed_25/bias/m/Read/ReadVariableOp,Nadam/conv1d_24/kernel/v/Read/ReadVariableOp*Nadam/conv1d_24/bias/v/Read/ReadVariableOp,Nadam/conv1d_25/kernel/v/Read/ReadVariableOp*Nadam/conv1d_25/bias/v/Read/ReadVariableOp5Nadam/gru_12/gru_cell_12/kernel/v/Read/ReadVariableOp?Nadam/gru_12/gru_cell_12/recurrent_kernel/v/Read/ReadVariableOp3Nadam/gru_12/gru_cell_12/bias/v/Read/ReadVariableOp6Nadam/time_distributed_24/kernel/v/Read/ReadVariableOp4Nadam/time_distributed_24/bias/v/Read/ReadVariableOp6Nadam/time_distributed_25/kernel/v/Read/ReadVariableOp4Nadam/time_distributed_25/bias/v/Read/ReadVariableOpConst*8
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_293297
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_24/kernelconv1d_24/biasconv1d_25/kernelconv1d_25/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachegru_12/gru_cell_12/kernel#gru_12/gru_cell_12/recurrent_kernelgru_12/gru_cell_12/biastime_distributed_24/kerneltime_distributed_24/biastime_distributed_25/kerneltime_distributed_25/biastotalcounttotal_1count_1Nadam/conv1d_24/kernel/mNadam/conv1d_24/bias/mNadam/conv1d_25/kernel/mNadam/conv1d_25/bias/m!Nadam/gru_12/gru_cell_12/kernel/m+Nadam/gru_12/gru_cell_12/recurrent_kernel/mNadam/gru_12/gru_cell_12/bias/m"Nadam/time_distributed_24/kernel/m Nadam/time_distributed_24/bias/m"Nadam/time_distributed_25/kernel/m Nadam/time_distributed_25/bias/mNadam/conv1d_24/kernel/vNadam/conv1d_24/bias/vNadam/conv1d_25/kernel/vNadam/conv1d_25/bias/v!Nadam/gru_12/gru_cell_12/kernel/v+Nadam/gru_12/gru_cell_12/recurrent_kernel/vNadam/gru_12/gru_cell_12/bias/v"Nadam/time_distributed_24/kernel/v Nadam/time_distributed_24/bias/v"Nadam/time_distributed_25/kernel/v Nadam/time_distributed_25/bias/v*7
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_293436??*
?
?
4__inference_time_distributed_25_layer_call_fn_292804

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2896372
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
?
b
F__inference_flatten_12_layer_call_and_return_conditional_losses_289807

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
?
.__inference_sequential_12_layer_call_fn_290677
conv1d_24_input
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
StatefulPartitionedCallStatefulPartitionedCallconv1d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
GPU2*0J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_2906252
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
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameconv1d_24_input
??
?	
while_body_292268
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_12_readvariableop_resource_0:	?H;
-while_gru_cell_12_readvariableop_3_resource_0:H?
-while_gru_cell_12_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_12_readvariableop_resource:	?H9
+while_gru_cell_12_readvariableop_3_resource:H=
+while_gru_cell_12_readvariableop_6_resource:H?? while/gru_cell_12/ReadVariableOp?"while/gru_cell_12/ReadVariableOp_1?"while/gru_cell_12/ReadVariableOp_2?"while/gru_cell_12/ReadVariableOp_3?"while/gru_cell_12/ReadVariableOp_4?"while/gru_cell_12/ReadVariableOp_5?"while/gru_cell_12/ReadVariableOp_6?"while/gru_cell_12/ReadVariableOp_7?"while/gru_cell_12/ReadVariableOp_8?
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
 while/gru_cell_12/ReadVariableOpReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02"
 while/gru_cell_12/ReadVariableOp?
%while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/gru_cell_12/strided_slice/stack?
'while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice/stack_1?
'while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell_12/strided_slice/stack_2?
while/gru_cell_12/strided_sliceStridedSlice(while/gru_cell_12/ReadVariableOp:value:0.while/gru_cell_12/strided_slice/stack:output:00while/gru_cell_12/strided_slice/stack_1:output:00while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2!
while/gru_cell_12/strided_slice?
while/gru_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0(while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul?
"while/gru_cell_12/ReadVariableOp_1ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_1?
'while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_1/stack?
)while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_1/stack_1?
)while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_1/stack_2?
!while/gru_cell_12/strided_slice_1StridedSlice*while/gru_cell_12/ReadVariableOp_1:value:00while/gru_cell_12/strided_slice_1/stack:output:02while/gru_cell_12/strided_slice_1/stack_1:output:02while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_1?
while/gru_cell_12/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_1?
"while/gru_cell_12/ReadVariableOp_2ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_2?
'while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_2/stack?
)while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_2/stack_1?
)while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_2/stack_2?
!while/gru_cell_12/strided_slice_2StridedSlice*while/gru_cell_12/ReadVariableOp_2:value:00while/gru_cell_12/strided_slice_2/stack:output:02while/gru_cell_12/strided_slice_2/stack_1:output:02while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_2?
while/gru_cell_12/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_2?
"while/gru_cell_12/ReadVariableOp_3ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_3?
'while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell_12/strided_slice_3/stack?
)while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_1?
)while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_2?
!while/gru_cell_12/strided_slice_3StridedSlice*while/gru_cell_12/ReadVariableOp_3:value:00while/gru_cell_12/strided_slice_3/stack:output:02while/gru_cell_12/strided_slice_3/stack_1:output:02while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2#
!while/gru_cell_12/strided_slice_3?
while/gru_cell_12/BiasAddBiasAdd"while/gru_cell_12/MatMul:product:0*while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd?
"while/gru_cell_12/ReadVariableOp_4ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_4?
'while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell_12/strided_slice_4/stack?
)while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02+
)while/gru_cell_12/strided_slice_4/stack_1?
)while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_4/stack_2?
!while/gru_cell_12/strided_slice_4StridedSlice*while/gru_cell_12/ReadVariableOp_4:value:00while/gru_cell_12/strided_slice_4/stack:output:02while/gru_cell_12/strided_slice_4/stack_1:output:02while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!while/gru_cell_12/strided_slice_4?
while/gru_cell_12/BiasAdd_1BiasAdd$while/gru_cell_12/MatMul_1:product:0*while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_1?
"while/gru_cell_12/ReadVariableOp_5ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_5?
'while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02)
'while/gru_cell_12/strided_slice_5/stack?
)while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_12/strided_slice_5/stack_1?
)while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_5/stack_2?
!while/gru_cell_12/strided_slice_5StridedSlice*while/gru_cell_12/ReadVariableOp_5:value:00while/gru_cell_12/strided_slice_5/stack:output:02while/gru_cell_12/strided_slice_5/stack_1:output:02while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2#
!while/gru_cell_12/strided_slice_5?
while/gru_cell_12/BiasAdd_2BiasAdd$while/gru_cell_12/MatMul_2:product:0*while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_2?
"while/gru_cell_12/ReadVariableOp_6ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_6?
'while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell_12/strided_slice_6/stack?
)while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/gru_cell_12/strided_slice_6/stack_1?
)while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_6/stack_2?
!while/gru_cell_12/strided_slice_6StridedSlice*while/gru_cell_12/ReadVariableOp_6:value:00while/gru_cell_12/strided_slice_6/stack:output:02while/gru_cell_12/strided_slice_6/stack_1:output:02while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_6?
while/gru_cell_12/MatMul_3MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_3?
"while/gru_cell_12/ReadVariableOp_7ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_7?
'while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_7/stack?
)while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_7/stack_1?
)while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_7/stack_2?
!while/gru_cell_12/strided_slice_7StridedSlice*while/gru_cell_12/ReadVariableOp_7:value:00while/gru_cell_12/strided_slice_7/stack:output:02while/gru_cell_12/strided_slice_7/stack_1:output:02while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_7?
while/gru_cell_12/MatMul_4MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_4?
while/gru_cell_12/addAddV2"while/gru_cell_12/BiasAdd:output:0$while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/addw
while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const{
while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_1?
while/gru_cell_12/MulMulwhile/gru_cell_12/add:z:0 while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul?
while/gru_cell_12/Add_1AddV2while/gru_cell_12/Mul:z:0"while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_1?
)while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/gru_cell_12/clip_by_value/Minimum/y?
'while/gru_cell_12/clip_by_value/MinimumMinimumwhile/gru_cell_12/Add_1:z:02while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2)
'while/gru_cell_12/clip_by_value/Minimum?
!while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/gru_cell_12/clip_by_value/y?
while/gru_cell_12/clip_by_valueMaximum+while/gru_cell_12/clip_by_value/Minimum:z:0*while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2!
while/gru_cell_12/clip_by_value?
while/gru_cell_12/add_2AddV2$while/gru_cell_12/BiasAdd_1:output:0$while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_2{
while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const_2{
while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_3?
while/gru_cell_12/Mul_1Mulwhile/gru_cell_12/add_2:z:0"while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul_1?
while/gru_cell_12/Add_3AddV2while/gru_cell_12/Mul_1:z:0"while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_3?
+while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+while/gru_cell_12/clip_by_value_1/Minimum/y?
)while/gru_cell_12/clip_by_value_1/MinimumMinimumwhile/gru_cell_12/Add_3:z:04while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)while/gru_cell_12/clip_by_value_1/Minimum?
#while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#while/gru_cell_12/clip_by_value_1/y?
!while/gru_cell_12/clip_by_value_1Maximum-while/gru_cell_12/clip_by_value_1/Minimum:z:0,while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2#
!while/gru_cell_12/clip_by_value_1?
while/gru_cell_12/mul_2Mul%while/gru_cell_12/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_2?
"while/gru_cell_12/ReadVariableOp_8ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_8?
'while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_8/stack?
)while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_8/stack_1?
)while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_8/stack_2?
!while/gru_cell_12/strided_slice_8StridedSlice*while/gru_cell_12/ReadVariableOp_8:value:00while/gru_cell_12/strided_slice_8/stack:output:02while/gru_cell_12/strided_slice_8/stack_1:output:02while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_8?
while/gru_cell_12/MatMul_5MatMulwhile/gru_cell_12/mul_2:z:0*while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_5?
while/gru_cell_12/add_4AddV2$while/gru_cell_12/BiasAdd_2:output:0$while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_4?
while/gru_cell_12/ReluReluwhile/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Relu?
while/gru_cell_12/mul_3Mul#while/gru_cell_12/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_3w
while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_12/sub/x?
while/gru_cell_12/subSub while/gru_cell_12/sub/x:output:0#while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/sub?
while/gru_cell_12/mul_4Mulwhile/gru_cell_12/sub:z:0$while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_4?
while/gru_cell_12/add_5AddV2while/gru_cell_12/mul_3:z:0while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_12/add_5:z:0*
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
while/Identity_4Identitywhile/gru_cell_12/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp!^while/gru_cell_12/ReadVariableOp#^while/gru_cell_12/ReadVariableOp_1#^while/gru_cell_12/ReadVariableOp_2#^while/gru_cell_12/ReadVariableOp_3#^while/gru_cell_12/ReadVariableOp_4#^while/gru_cell_12/ReadVariableOp_5#^while/gru_cell_12/ReadVariableOp_6#^while/gru_cell_12/ReadVariableOp_7#^while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"\
+while_gru_cell_12_readvariableop_3_resource-while_gru_cell_12_readvariableop_3_resource_0"\
+while_gru_cell_12_readvariableop_6_resource-while_gru_cell_12_readvariableop_6_resource_0"X
)while_gru_cell_12_readvariableop_resource+while_gru_cell_12_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2D
 while/gru_cell_12/ReadVariableOp while/gru_cell_12/ReadVariableOp2H
"while/gru_cell_12/ReadVariableOp_1"while/gru_cell_12/ReadVariableOp_12H
"while/gru_cell_12/ReadVariableOp_2"while/gru_cell_12/ReadVariableOp_22H
"while/gru_cell_12/ReadVariableOp_3"while/gru_cell_12/ReadVariableOp_32H
"while/gru_cell_12/ReadVariableOp_4"while/gru_cell_12/ReadVariableOp_42H
"while/gru_cell_12/ReadVariableOp_5"while/gru_cell_12/ReadVariableOp_52H
"while/gru_cell_12/ReadVariableOp_6"while/gru_cell_12/ReadVariableOp_62H
"while/gru_cell_12/ReadVariableOp_7"while/gru_cell_12/ReadVariableOp_72H
"while/gru_cell_12/ReadVariableOp_8"while/gru_cell_12/ReadVariableOp_8: 
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
?
h
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_288769

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
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_290122

inputs9
'dense_25_matmul_readvariableop_resource: 6
(dense_25_biasadd_readvariableop_resource:
identity??dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOpo
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
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMulReshape:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_25/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_289498

inputs!
dense_24_289488: 
dense_24_289490: 
identity?? dense_24/StatefulPartitionedCallD
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
 dense_24/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_24_289488dense_24_289490*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2894872"
 dense_24/StatefulPartitionedCallq
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
	Reshape_1Reshape)dense_24/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityq
NoOpNoOp!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_289637

inputs!
dense_25_289627: 
dense_25_289629:
identity?? dense_25/StatefulPartitionedCallD
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
 dense_25/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_25_289627dense_25_289629*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_2896262"
 dense_25/StatefulPartitionedCallq
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
	Reshape_1Reshape)dense_25/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityq
NoOpNoOp!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_290213

inputs9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource: 
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOpo
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
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulReshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense_24/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_289799

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
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_290181

inputs9
'dense_25_matmul_readvariableop_resource: 6
(dense_25_biasadd_readvariableop_resource:
identity??dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOpo
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
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMulReshape:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_25/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?3
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_290755
conv1d_24_input&
conv1d_24_290719: 
conv1d_24_290721: &
conv1d_25_290724: @
conv1d_25_290726:@ 
gru_12_290732:	?H
gru_12_290734:H
gru_12_290736:H,
time_distributed_24_290740: (
time_distributed_24_290742: ,
time_distributed_25_290747: (
time_distributed_25_290749:
identity??!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?gru_12/StatefulPartitionedCall?+time_distributed_24/StatefulPartitionedCall?+time_distributed_25/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCallconv1d_24_inputconv1d_24_290719conv1d_24_290721*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2897642#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_290724conv1d_25_290726*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2897862#
!conv1d_25/StatefulPartitionedCall?
 max_pooling1d_12/PartitionedCallPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_2897992"
 max_pooling1d_12/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling1d_12/PartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_2898072
flatten_12/PartitionedCall?
 repeat_vector_12/PartitionedCallPartitionedCall#flatten_12/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_2898162"
 repeat_vector_12/PartitionedCall?
gru_12/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_12/PartitionedCall:output:0gru_12_290732gru_12_290734gru_12_290736*
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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2905122 
gru_12/StatefulPartitionedCall?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall'gru_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2902402$
"dropout_12/StatefulPartitionedCall?
+time_distributed_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0time_distributed_24_290740time_distributed_24_290742*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2902132-
+time_distributed_24/StatefulPartitionedCall?
!time_distributed_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!time_distributed_24/Reshape/shape?
time_distributed_24/ReshapeReshape+dropout_12/StatefulPartitionedCall:output:0*time_distributed_24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape?
+time_distributed_25/StatefulPartitionedCallStatefulPartitionedCall4time_distributed_24/StatefulPartitionedCall:output:0time_distributed_25_290747time_distributed_25_290749*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2901812-
+time_distributed_25/StatefulPartitionedCall?
!time_distributed_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!time_distributed_25/Reshape/shape?
time_distributed_25/ReshapeReshape4time_distributed_24/StatefulPartitionedCall:output:0*time_distributed_25/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape?
IdentityIdentity4time_distributed_25/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall^gru_12/StatefulPartitionedCall,^time_distributed_24/StatefulPartitionedCall,^time_distributed_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2Z
+time_distributed_24/StatefulPartitionedCall+time_distributed_24/StatefulPartitionedCall2Z
+time_distributed_25/StatefulPartitionedCall+time_distributed_25/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameconv1d_24_input
??
?	
while_body_292524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_12_readvariableop_resource_0:	?H;
-while_gru_cell_12_readvariableop_3_resource_0:H?
-while_gru_cell_12_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_12_readvariableop_resource:	?H9
+while_gru_cell_12_readvariableop_3_resource:H=
+while_gru_cell_12_readvariableop_6_resource:H?? while/gru_cell_12/ReadVariableOp?"while/gru_cell_12/ReadVariableOp_1?"while/gru_cell_12/ReadVariableOp_2?"while/gru_cell_12/ReadVariableOp_3?"while/gru_cell_12/ReadVariableOp_4?"while/gru_cell_12/ReadVariableOp_5?"while/gru_cell_12/ReadVariableOp_6?"while/gru_cell_12/ReadVariableOp_7?"while/gru_cell_12/ReadVariableOp_8?
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
 while/gru_cell_12/ReadVariableOpReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02"
 while/gru_cell_12/ReadVariableOp?
%while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/gru_cell_12/strided_slice/stack?
'while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice/stack_1?
'while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell_12/strided_slice/stack_2?
while/gru_cell_12/strided_sliceStridedSlice(while/gru_cell_12/ReadVariableOp:value:0.while/gru_cell_12/strided_slice/stack:output:00while/gru_cell_12/strided_slice/stack_1:output:00while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2!
while/gru_cell_12/strided_slice?
while/gru_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0(while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul?
"while/gru_cell_12/ReadVariableOp_1ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_1?
'while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_1/stack?
)while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_1/stack_1?
)while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_1/stack_2?
!while/gru_cell_12/strided_slice_1StridedSlice*while/gru_cell_12/ReadVariableOp_1:value:00while/gru_cell_12/strided_slice_1/stack:output:02while/gru_cell_12/strided_slice_1/stack_1:output:02while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_1?
while/gru_cell_12/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_1?
"while/gru_cell_12/ReadVariableOp_2ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_2?
'while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_2/stack?
)while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_2/stack_1?
)while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_2/stack_2?
!while/gru_cell_12/strided_slice_2StridedSlice*while/gru_cell_12/ReadVariableOp_2:value:00while/gru_cell_12/strided_slice_2/stack:output:02while/gru_cell_12/strided_slice_2/stack_1:output:02while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_2?
while/gru_cell_12/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_2?
"while/gru_cell_12/ReadVariableOp_3ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_3?
'while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell_12/strided_slice_3/stack?
)while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_1?
)while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_2?
!while/gru_cell_12/strided_slice_3StridedSlice*while/gru_cell_12/ReadVariableOp_3:value:00while/gru_cell_12/strided_slice_3/stack:output:02while/gru_cell_12/strided_slice_3/stack_1:output:02while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2#
!while/gru_cell_12/strided_slice_3?
while/gru_cell_12/BiasAddBiasAdd"while/gru_cell_12/MatMul:product:0*while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd?
"while/gru_cell_12/ReadVariableOp_4ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_4?
'while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell_12/strided_slice_4/stack?
)while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02+
)while/gru_cell_12/strided_slice_4/stack_1?
)while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_4/stack_2?
!while/gru_cell_12/strided_slice_4StridedSlice*while/gru_cell_12/ReadVariableOp_4:value:00while/gru_cell_12/strided_slice_4/stack:output:02while/gru_cell_12/strided_slice_4/stack_1:output:02while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!while/gru_cell_12/strided_slice_4?
while/gru_cell_12/BiasAdd_1BiasAdd$while/gru_cell_12/MatMul_1:product:0*while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_1?
"while/gru_cell_12/ReadVariableOp_5ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_5?
'while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02)
'while/gru_cell_12/strided_slice_5/stack?
)while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_12/strided_slice_5/stack_1?
)while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_5/stack_2?
!while/gru_cell_12/strided_slice_5StridedSlice*while/gru_cell_12/ReadVariableOp_5:value:00while/gru_cell_12/strided_slice_5/stack:output:02while/gru_cell_12/strided_slice_5/stack_1:output:02while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2#
!while/gru_cell_12/strided_slice_5?
while/gru_cell_12/BiasAdd_2BiasAdd$while/gru_cell_12/MatMul_2:product:0*while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_2?
"while/gru_cell_12/ReadVariableOp_6ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_6?
'while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell_12/strided_slice_6/stack?
)while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/gru_cell_12/strided_slice_6/stack_1?
)while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_6/stack_2?
!while/gru_cell_12/strided_slice_6StridedSlice*while/gru_cell_12/ReadVariableOp_6:value:00while/gru_cell_12/strided_slice_6/stack:output:02while/gru_cell_12/strided_slice_6/stack_1:output:02while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_6?
while/gru_cell_12/MatMul_3MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_3?
"while/gru_cell_12/ReadVariableOp_7ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_7?
'while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_7/stack?
)while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_7/stack_1?
)while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_7/stack_2?
!while/gru_cell_12/strided_slice_7StridedSlice*while/gru_cell_12/ReadVariableOp_7:value:00while/gru_cell_12/strided_slice_7/stack:output:02while/gru_cell_12/strided_slice_7/stack_1:output:02while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_7?
while/gru_cell_12/MatMul_4MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_4?
while/gru_cell_12/addAddV2"while/gru_cell_12/BiasAdd:output:0$while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/addw
while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const{
while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_1?
while/gru_cell_12/MulMulwhile/gru_cell_12/add:z:0 while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul?
while/gru_cell_12/Add_1AddV2while/gru_cell_12/Mul:z:0"while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_1?
)while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/gru_cell_12/clip_by_value/Minimum/y?
'while/gru_cell_12/clip_by_value/MinimumMinimumwhile/gru_cell_12/Add_1:z:02while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2)
'while/gru_cell_12/clip_by_value/Minimum?
!while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/gru_cell_12/clip_by_value/y?
while/gru_cell_12/clip_by_valueMaximum+while/gru_cell_12/clip_by_value/Minimum:z:0*while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2!
while/gru_cell_12/clip_by_value?
while/gru_cell_12/add_2AddV2$while/gru_cell_12/BiasAdd_1:output:0$while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_2{
while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const_2{
while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_3?
while/gru_cell_12/Mul_1Mulwhile/gru_cell_12/add_2:z:0"while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul_1?
while/gru_cell_12/Add_3AddV2while/gru_cell_12/Mul_1:z:0"while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_3?
+while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+while/gru_cell_12/clip_by_value_1/Minimum/y?
)while/gru_cell_12/clip_by_value_1/MinimumMinimumwhile/gru_cell_12/Add_3:z:04while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)while/gru_cell_12/clip_by_value_1/Minimum?
#while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#while/gru_cell_12/clip_by_value_1/y?
!while/gru_cell_12/clip_by_value_1Maximum-while/gru_cell_12/clip_by_value_1/Minimum:z:0,while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2#
!while/gru_cell_12/clip_by_value_1?
while/gru_cell_12/mul_2Mul%while/gru_cell_12/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_2?
"while/gru_cell_12/ReadVariableOp_8ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_8?
'while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_8/stack?
)while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_8/stack_1?
)while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_8/stack_2?
!while/gru_cell_12/strided_slice_8StridedSlice*while/gru_cell_12/ReadVariableOp_8:value:00while/gru_cell_12/strided_slice_8/stack:output:02while/gru_cell_12/strided_slice_8/stack_1:output:02while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_8?
while/gru_cell_12/MatMul_5MatMulwhile/gru_cell_12/mul_2:z:0*while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_5?
while/gru_cell_12/add_4AddV2$while/gru_cell_12/BiasAdd_2:output:0$while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_4?
while/gru_cell_12/ReluReluwhile/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Relu?
while/gru_cell_12/mul_3Mul#while/gru_cell_12/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_3w
while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_12/sub/x?
while/gru_cell_12/subSub while/gru_cell_12/sub/x:output:0#while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/sub?
while/gru_cell_12/mul_4Mulwhile/gru_cell_12/sub:z:0$while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_4?
while/gru_cell_12/add_5AddV2while/gru_cell_12/mul_3:z:0while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_12/add_5:z:0*
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
while/Identity_4Identitywhile/gru_cell_12/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp!^while/gru_cell_12/ReadVariableOp#^while/gru_cell_12/ReadVariableOp_1#^while/gru_cell_12/ReadVariableOp_2#^while/gru_cell_12/ReadVariableOp_3#^while/gru_cell_12/ReadVariableOp_4#^while/gru_cell_12/ReadVariableOp_5#^while/gru_cell_12/ReadVariableOp_6#^while/gru_cell_12/ReadVariableOp_7#^while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"\
+while_gru_cell_12_readvariableop_3_resource-while_gru_cell_12_readvariableop_3_resource_0"\
+while_gru_cell_12_readvariableop_6_resource-while_gru_cell_12_readvariableop_6_resource_0"X
)while_gru_cell_12_readvariableop_resource+while_gru_cell_12_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2D
 while/gru_cell_12/ReadVariableOp while/gru_cell_12/ReadVariableOp2H
"while/gru_cell_12/ReadVariableOp_1"while/gru_cell_12/ReadVariableOp_12H
"while/gru_cell_12/ReadVariableOp_2"while/gru_cell_12/ReadVariableOp_22H
"while/gru_cell_12/ReadVariableOp_3"while/gru_cell_12/ReadVariableOp_32H
"while/gru_cell_12/ReadVariableOp_4"while/gru_cell_12/ReadVariableOp_42H
"while/gru_cell_12/ReadVariableOp_5"while/gru_cell_12/ReadVariableOp_52H
"while/gru_cell_12/ReadVariableOp_6"while/gru_cell_12/ReadVariableOp_62H
"while/gru_cell_12/ReadVariableOp_7"while/gru_cell_12/ReadVariableOp_72H
"while/gru_cell_12/ReadVariableOp_8"while/gru_cell_12/ReadVariableOp_8: 
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
B__inference_gru_12_layer_call_and_return_conditional_losses_288985

inputs%
gru_cell_12_288910:	?H 
gru_cell_12_288912:H$
gru_cell_12_288914:H
identity??#gru_cell_12/StatefulPartitionedCall?whileD
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
#gru_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_12_288910gru_cell_12_288912gru_cell_12_288914*
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
GPU2*0J 8? *P
fKRI
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_2889092%
#gru_cell_12/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_12_288910gru_cell_12_288912gru_cell_12_288914*
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
bodyR
while_body_288922*
condR
while_cond_288921*8
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

Identity|
NoOpNoOp$^gru_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_12/StatefulPartitionedCall#gru_cell_12/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292781

inputs9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource: 
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOpo
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
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulReshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense_24/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_290625

inputs&
conv1d_24_290589: 
conv1d_24_290591: &
conv1d_25_290594: @
conv1d_25_290596:@ 
gru_12_290602:	?H
gru_12_290604:H
gru_12_290606:H,
time_distributed_24_290610: (
time_distributed_24_290612: ,
time_distributed_25_290617: (
time_distributed_25_290619:
identity??!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?gru_12/StatefulPartitionedCall?+time_distributed_24/StatefulPartitionedCall?+time_distributed_25/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_24_290589conv1d_24_290591*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2897642#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_290594conv1d_25_290596*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2897862#
!conv1d_25/StatefulPartitionedCall?
 max_pooling1d_12/PartitionedCallPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_2897992"
 max_pooling1d_12/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling1d_12/PartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_2898072
flatten_12/PartitionedCall?
 repeat_vector_12/PartitionedCallPartitionedCall#flatten_12/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_2898162"
 repeat_vector_12/PartitionedCall?
gru_12/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_12/PartitionedCall:output:0gru_12_290602gru_12_290604gru_12_290606*
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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2905122 
gru_12/StatefulPartitionedCall?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall'gru_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2902402$
"dropout_12/StatefulPartitionedCall?
+time_distributed_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0time_distributed_24_290610time_distributed_24_290612*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2902132-
+time_distributed_24/StatefulPartitionedCall?
!time_distributed_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!time_distributed_24/Reshape/shape?
time_distributed_24/ReshapeReshape+dropout_12/StatefulPartitionedCall:output:0*time_distributed_24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape?
+time_distributed_25/StatefulPartitionedCallStatefulPartitionedCall4time_distributed_24/StatefulPartitionedCall:output:0time_distributed_25_290617time_distributed_25_290619*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2901812-
+time_distributed_25/StatefulPartitionedCall?
!time_distributed_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!time_distributed_25/Reshape/shape?
time_distributed_25/ReshapeReshape4time_distributed_24/StatefulPartitionedCall:output:0*time_distributed_25/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape?
IdentityIdentity4time_distributed_25/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall^gru_12/StatefulPartitionedCall,^time_distributed_24/StatefulPartitionedCall,^time_distributed_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2Z
+time_distributed_24/StatefulPartitionedCall+time_distributed_24/StatefulPartitionedCall2Z
+time_distributed_25/StatefulPartitionedCall+time_distributed_25/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_291159

inputsK
5conv1d_24_conv1d_expanddims_1_readvariableop_resource: 7
)conv1d_24_biasadd_readvariableop_resource: K
5conv1d_25_conv1d_expanddims_1_readvariableop_resource: @7
)conv1d_25_biasadd_readvariableop_resource:@=
*gru_12_gru_cell_12_readvariableop_resource:	?H:
,gru_12_gru_cell_12_readvariableop_3_resource:H>
,gru_12_gru_cell_12_readvariableop_6_resource:HM
;time_distributed_24_dense_24_matmul_readvariableop_resource: J
<time_distributed_24_dense_24_biasadd_readvariableop_resource: M
;time_distributed_25_dense_25_matmul_readvariableop_resource: J
<time_distributed_25_dense_25_biasadd_readvariableop_resource:
identity?? conv1d_24/BiasAdd/ReadVariableOp?,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp? conv1d_25/BiasAdd/ReadVariableOp?,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?!gru_12/gru_cell_12/ReadVariableOp?#gru_12/gru_cell_12/ReadVariableOp_1?#gru_12/gru_cell_12/ReadVariableOp_2?#gru_12/gru_cell_12/ReadVariableOp_3?#gru_12/gru_cell_12/ReadVariableOp_4?#gru_12/gru_cell_12/ReadVariableOp_5?#gru_12/gru_cell_12/ReadVariableOp_6?#gru_12/gru_cell_12/ReadVariableOp_7?#gru_12/gru_cell_12/ReadVariableOp_8?gru_12/while?3time_distributed_24/dense_24/BiasAdd/ReadVariableOp?2time_distributed_24/dense_24/MatMul/ReadVariableOp?3time_distributed_25/dense_25/BiasAdd/ReadVariableOp?2time_distributed_25/dense_25/MatMul/ReadVariableOp?
conv1d_24/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_24/conv1d/ExpandDims/dim?
conv1d_24/conv1d/ExpandDims
ExpandDimsinputs(conv1d_24/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_24/conv1d/ExpandDims?
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_24_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02.
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_24/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_24/conv1d/ExpandDims_1/dim?
conv1d_24/conv1d/ExpandDims_1
ExpandDims4conv1d_24/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_24/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_24/conv1d/ExpandDims_1?
conv1d_24/conv1dConv2D$conv1d_24/conv1d/ExpandDims:output:0&conv1d_24/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d_24/conv1d?
conv1d_24/conv1d/SqueezeSqueezeconv1d_24/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_24/conv1d/Squeeze?
 conv1d_24/BiasAdd/ReadVariableOpReadVariableOp)conv1d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_24/BiasAdd/ReadVariableOp?
conv1d_24/BiasAddBiasAdd!conv1d_24/conv1d/Squeeze:output:0(conv1d_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d_24/BiasAddz
conv1d_24/ReluReluconv1d_24/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
conv1d_24/Relu?
conv1d_25/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_25/conv1d/ExpandDims/dim?
conv1d_25/conv1d/ExpandDims
ExpandDimsconv1d_24/Relu:activations:0(conv1d_25/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d_25/conv1d/ExpandDims?
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_25_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02.
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_25/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_25/conv1d/ExpandDims_1/dim?
conv1d_25/conv1d/ExpandDims_1
ExpandDims4conv1d_25/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_25/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_25/conv1d/ExpandDims_1?
conv1d_25/conv1dConv2D$conv1d_25/conv1d/ExpandDims:output:0&conv1d_25/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_25/conv1d?
conv1d_25/conv1d/SqueezeSqueezeconv1d_25/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_25/conv1d/Squeeze?
 conv1d_25/BiasAdd/ReadVariableOpReadVariableOp)conv1d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_25/BiasAdd/ReadVariableOp?
conv1d_25/BiasAddBiasAdd!conv1d_25/conv1d/Squeeze:output:0(conv1d_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/BiasAddz
conv1d_25/ReluReluconv1d_25/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/Relu?
max_pooling1d_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_12/ExpandDims/dim?
max_pooling1d_12/ExpandDims
ExpandDimsconv1d_25/Relu:activations:0(max_pooling1d_12/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
max_pooling1d_12/ExpandDims?
max_pooling1d_12/MaxPoolMaxPool$max_pooling1d_12/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_12/MaxPool?
max_pooling1d_12/SqueezeSqueeze!max_pooling1d_12/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
max_pooling1d_12/Squeezeu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_12/Const?
flatten_12/ReshapeReshape!max_pooling1d_12/Squeeze:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
repeat_vector_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_12/ExpandDims/dim?
repeat_vector_12/ExpandDims
ExpandDimsflatten_12/Reshape:output:0(repeat_vector_12/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector_12/ExpandDims?
repeat_vector_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
repeat_vector_12/stack?
repeat_vector_12/TileTile$repeat_vector_12/ExpandDims:output:0repeat_vector_12/stack:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector_12/Tilej
gru_12/ShapeShaperepeat_vector_12/Tile:output:0*
T0*
_output_shapes
:2
gru_12/Shape?
gru_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_12/strided_slice/stack?
gru_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_12/strided_slice/stack_1?
gru_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_12/strided_slice/stack_2?
gru_12/strided_sliceStridedSlicegru_12/Shape:output:0#gru_12/strided_slice/stack:output:0%gru_12/strided_slice/stack_1:output:0%gru_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_12/strided_slicej
gru_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_12/zeros/mul/y?
gru_12/zeros/mulMulgru_12/strided_slice:output:0gru_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_12/zeros/mulm
gru_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_12/zeros/Less/y?
gru_12/zeros/LessLessgru_12/zeros/mul:z:0gru_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_12/zeros/Lessp
gru_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_12/zeros/packed/1?
gru_12/zeros/packedPackgru_12/strided_slice:output:0gru_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_12/zeros/packedm
gru_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_12/zeros/Const?
gru_12/zerosFillgru_12/zeros/packed:output:0gru_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_12/zeros?
gru_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_12/transpose/perm?
gru_12/transpose	Transposerepeat_vector_12/Tile:output:0gru_12/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_12/transposed
gru_12/Shape_1Shapegru_12/transpose:y:0*
T0*
_output_shapes
:2
gru_12/Shape_1?
gru_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_12/strided_slice_1/stack?
gru_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_1/stack_1?
gru_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_1/stack_2?
gru_12/strided_slice_1StridedSlicegru_12/Shape_1:output:0%gru_12/strided_slice_1/stack:output:0'gru_12/strided_slice_1/stack_1:output:0'gru_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_12/strided_slice_1?
"gru_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_12/TensorArrayV2/element_shape?
gru_12/TensorArrayV2TensorListReserve+gru_12/TensorArrayV2/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_12/TensorArrayV2?
<gru_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<gru_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_12/transpose:y:0Egru_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_12/TensorArrayUnstack/TensorListFromTensor?
gru_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_12/strided_slice_2/stack?
gru_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_2/stack_1?
gru_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_2/stack_2?
gru_12/strided_slice_2StridedSlicegru_12/transpose:y:0%gru_12/strided_slice_2/stack:output:0'gru_12/strided_slice_2/stack_1:output:0'gru_12/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_12/strided_slice_2?
!gru_12/gru_cell_12/ReadVariableOpReadVariableOp*gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02#
!gru_12/gru_cell_12/ReadVariableOp?
&gru_12/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru_12/gru_cell_12/strided_slice/stack?
(gru_12/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(gru_12/gru_cell_12/strided_slice/stack_1?
(gru_12/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_12/gru_cell_12/strided_slice/stack_2?
 gru_12/gru_cell_12/strided_sliceStridedSlice)gru_12/gru_cell_12/ReadVariableOp:value:0/gru_12/gru_cell_12/strided_slice/stack:output:01gru_12/gru_cell_12/strided_slice/stack_1:output:01gru_12/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2"
 gru_12/gru_cell_12/strided_slice?
gru_12/gru_cell_12/MatMulMatMulgru_12/strided_slice_2:output:0)gru_12/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul?
#gru_12/gru_cell_12/ReadVariableOp_1ReadVariableOp*gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_1?
(gru_12/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru_12/gru_cell_12/strided_slice_1/stack?
*gru_12/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru_12/gru_cell_12/strided_slice_1/stack_1?
*gru_12/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_1/stack_2?
"gru_12/gru_cell_12/strided_slice_1StridedSlice+gru_12/gru_cell_12/ReadVariableOp_1:value:01gru_12/gru_cell_12/strided_slice_1/stack:output:03gru_12/gru_cell_12/strided_slice_1/stack_1:output:03gru_12/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_1?
gru_12/gru_cell_12/MatMul_1MatMulgru_12/strided_slice_2:output:0+gru_12/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_1?
#gru_12/gru_cell_12/ReadVariableOp_2ReadVariableOp*gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_2?
(gru_12/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru_12/gru_cell_12/strided_slice_2/stack?
*gru_12/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_12/gru_cell_12/strided_slice_2/stack_1?
*gru_12/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_2/stack_2?
"gru_12/gru_cell_12/strided_slice_2StridedSlice+gru_12/gru_cell_12/ReadVariableOp_2:value:01gru_12/gru_cell_12/strided_slice_2/stack:output:03gru_12/gru_cell_12/strided_slice_2/stack_1:output:03gru_12/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_2?
gru_12/gru_cell_12/MatMul_2MatMulgru_12/strided_slice_2:output:0+gru_12/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_2?
#gru_12/gru_cell_12/ReadVariableOp_3ReadVariableOp,gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_3?
(gru_12/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru_12/gru_cell_12/strided_slice_3/stack?
*gru_12/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_3/stack_1?
*gru_12/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_3/stack_2?
"gru_12/gru_cell_12/strided_slice_3StridedSlice+gru_12/gru_cell_12/ReadVariableOp_3:value:01gru_12/gru_cell_12/strided_slice_3/stack:output:03gru_12/gru_cell_12/strided_slice_3/stack_1:output:03gru_12/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"gru_12/gru_cell_12/strided_slice_3?
gru_12/gru_cell_12/BiasAddBiasAdd#gru_12/gru_cell_12/MatMul:product:0+gru_12/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/BiasAdd?
#gru_12/gru_cell_12/ReadVariableOp_4ReadVariableOp,gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_4?
(gru_12/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(gru_12/gru_cell_12/strided_slice_4/stack?
*gru_12/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02,
*gru_12/gru_cell_12/strided_slice_4/stack_1?
*gru_12/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_4/stack_2?
"gru_12/gru_cell_12/strided_slice_4StridedSlice+gru_12/gru_cell_12/ReadVariableOp_4:value:01gru_12/gru_cell_12/strided_slice_4/stack:output:03gru_12/gru_cell_12/strided_slice_4/stack_1:output:03gru_12/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2$
"gru_12/gru_cell_12/strided_slice_4?
gru_12/gru_cell_12/BiasAdd_1BiasAdd%gru_12/gru_cell_12/MatMul_1:product:0+gru_12/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/BiasAdd_1?
#gru_12/gru_cell_12/ReadVariableOp_5ReadVariableOp,gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_5?
(gru_12/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02*
(gru_12/gru_cell_12/strided_slice_5/stack?
*gru_12/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru_12/gru_cell_12/strided_slice_5/stack_1?
*gru_12/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_5/stack_2?
"gru_12/gru_cell_12/strided_slice_5StridedSlice+gru_12/gru_cell_12/ReadVariableOp_5:value:01gru_12/gru_cell_12/strided_slice_5/stack:output:03gru_12/gru_cell_12/strided_slice_5/stack_1:output:03gru_12/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2$
"gru_12/gru_cell_12/strided_slice_5?
gru_12/gru_cell_12/BiasAdd_2BiasAdd%gru_12/gru_cell_12/MatMul_2:product:0+gru_12/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/BiasAdd_2?
#gru_12/gru_cell_12/ReadVariableOp_6ReadVariableOp,gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_6?
(gru_12/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru_12/gru_cell_12/strided_slice_6/stack?
*gru_12/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*gru_12/gru_cell_12/strided_slice_6/stack_1?
*gru_12/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_6/stack_2?
"gru_12/gru_cell_12/strided_slice_6StridedSlice+gru_12/gru_cell_12/ReadVariableOp_6:value:01gru_12/gru_cell_12/strided_slice_6/stack:output:03gru_12/gru_cell_12/strided_slice_6/stack_1:output:03gru_12/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_6?
gru_12/gru_cell_12/MatMul_3MatMulgru_12/zeros:output:0+gru_12/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_3?
#gru_12/gru_cell_12/ReadVariableOp_7ReadVariableOp,gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_7?
(gru_12/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru_12/gru_cell_12/strided_slice_7/stack?
*gru_12/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru_12/gru_cell_12/strided_slice_7/stack_1?
*gru_12/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_7/stack_2?
"gru_12/gru_cell_12/strided_slice_7StridedSlice+gru_12/gru_cell_12/ReadVariableOp_7:value:01gru_12/gru_cell_12/strided_slice_7/stack:output:03gru_12/gru_cell_12/strided_slice_7/stack_1:output:03gru_12/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_7?
gru_12/gru_cell_12/MatMul_4MatMulgru_12/zeros:output:0+gru_12/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_4?
gru_12/gru_cell_12/addAddV2#gru_12/gru_cell_12/BiasAdd:output:0%gru_12/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/addy
gru_12/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_12/gru_cell_12/Const}
gru_12/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_12/gru_cell_12/Const_1?
gru_12/gru_cell_12/MulMulgru_12/gru_cell_12/add:z:0!gru_12/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Mul?
gru_12/gru_cell_12/Add_1AddV2gru_12/gru_cell_12/Mul:z:0#gru_12/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Add_1?
*gru_12/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*gru_12/gru_cell_12/clip_by_value/Minimum/y?
(gru_12/gru_cell_12/clip_by_value/MinimumMinimumgru_12/gru_cell_12/Add_1:z:03gru_12/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(gru_12/gru_cell_12/clip_by_value/Minimum?
"gru_12/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"gru_12/gru_cell_12/clip_by_value/y?
 gru_12/gru_cell_12/clip_by_valueMaximum,gru_12/gru_cell_12/clip_by_value/Minimum:z:0+gru_12/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_12/gru_cell_12/clip_by_value?
gru_12/gru_cell_12/add_2AddV2%gru_12/gru_cell_12/BiasAdd_1:output:0%gru_12/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/add_2}
gru_12/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_12/gru_cell_12/Const_2}
gru_12/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_12/gru_cell_12/Const_3?
gru_12/gru_cell_12/Mul_1Mulgru_12/gru_cell_12/add_2:z:0#gru_12/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Mul_1?
gru_12/gru_cell_12/Add_3AddV2gru_12/gru_cell_12/Mul_1:z:0#gru_12/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Add_3?
,gru_12/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,gru_12/gru_cell_12/clip_by_value_1/Minimum/y?
*gru_12/gru_cell_12/clip_by_value_1/MinimumMinimumgru_12/gru_cell_12/Add_3:z:05gru_12/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2,
*gru_12/gru_cell_12/clip_by_value_1/Minimum?
$gru_12/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$gru_12/gru_cell_12/clip_by_value_1/y?
"gru_12/gru_cell_12/clip_by_value_1Maximum.gru_12/gru_cell_12/clip_by_value_1/Minimum:z:0-gru_12/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2$
"gru_12/gru_cell_12/clip_by_value_1?
gru_12/gru_cell_12/mul_2Mul&gru_12/gru_cell_12/clip_by_value_1:z:0gru_12/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/mul_2?
#gru_12/gru_cell_12/ReadVariableOp_8ReadVariableOp,gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_8?
(gru_12/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru_12/gru_cell_12/strided_slice_8/stack?
*gru_12/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_12/gru_cell_12/strided_slice_8/stack_1?
*gru_12/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_8/stack_2?
"gru_12/gru_cell_12/strided_slice_8StridedSlice+gru_12/gru_cell_12/ReadVariableOp_8:value:01gru_12/gru_cell_12/strided_slice_8/stack:output:03gru_12/gru_cell_12/strided_slice_8/stack_1:output:03gru_12/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_8?
gru_12/gru_cell_12/MatMul_5MatMulgru_12/gru_cell_12/mul_2:z:0+gru_12/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_5?
gru_12/gru_cell_12/add_4AddV2%gru_12/gru_cell_12/BiasAdd_2:output:0%gru_12/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/add_4?
gru_12/gru_cell_12/ReluRelugru_12/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Relu?
gru_12/gru_cell_12/mul_3Mul$gru_12/gru_cell_12/clip_by_value:z:0gru_12/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/mul_3y
gru_12/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_12/gru_cell_12/sub/x?
gru_12/gru_cell_12/subSub!gru_12/gru_cell_12/sub/x:output:0$gru_12/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/sub?
gru_12/gru_cell_12/mul_4Mulgru_12/gru_cell_12/sub:z:0%gru_12/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/mul_4?
gru_12/gru_cell_12/add_5AddV2gru_12/gru_cell_12/mul_3:z:0gru_12/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/add_5?
$gru_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_12/TensorArrayV2_1/element_shape?
gru_12/TensorArrayV2_1TensorListReserve-gru_12/TensorArrayV2_1/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_12/TensorArrayV2_1\
gru_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_12/time?
gru_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_12/while/maximum_iterationsx
gru_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_12/while/loop_counter?
gru_12/whileWhile"gru_12/while/loop_counter:output:0(gru_12/while/maximum_iterations:output:0gru_12/time:output:0gru_12/TensorArrayV2_1:handle:0gru_12/zeros:output:0gru_12/strided_slice_1:output:0>gru_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_12_gru_cell_12_readvariableop_resource,gru_12_gru_cell_12_readvariableop_3_resource,gru_12_gru_cell_12_readvariableop_6_resource*
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
_stateful_parallelism( *$
bodyR
gru_12_while_body_290996*$
condR
gru_12_while_cond_290995*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
gru_12/while?
7gru_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_12/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_12/TensorArrayV2Stack/TensorListStackTensorListStackgru_12/while:output:3@gru_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02+
)gru_12/TensorArrayV2Stack/TensorListStack?
gru_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_12/strided_slice_3/stack?
gru_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_12/strided_slice_3/stack_1?
gru_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_3/stack_2?
gru_12/strided_slice_3StridedSlice2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0%gru_12/strided_slice_3/stack:output:0'gru_12/strided_slice_3/stack_1:output:0'gru_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_12/strided_slice_3?
gru_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_12/transpose_1/perm?
gru_12/transpose_1	Transpose2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0 gru_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_12/transpose_1?
dropout_12/IdentityIdentitygru_12/transpose_1:y:0*
T0*+
_output_shapes
:?????????2
dropout_12/Identity?
!time_distributed_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!time_distributed_24/Reshape/shape?
time_distributed_24/ReshapeReshapedropout_12/Identity:output:0*time_distributed_24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape?
2time_distributed_24/dense_24/MatMul/ReadVariableOpReadVariableOp;time_distributed_24_dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype024
2time_distributed_24/dense_24/MatMul/ReadVariableOp?
#time_distributed_24/dense_24/MatMulMatMul$time_distributed_24/Reshape:output:0:time_distributed_24/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#time_distributed_24/dense_24/MatMul?
3time_distributed_24/dense_24/BiasAdd/ReadVariableOpReadVariableOp<time_distributed_24_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3time_distributed_24/dense_24/BiasAdd/ReadVariableOp?
$time_distributed_24/dense_24/BiasAddBiasAdd-time_distributed_24/dense_24/MatMul:product:0;time_distributed_24/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$time_distributed_24/dense_24/BiasAdd?
#time_distributed_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2%
#time_distributed_24/Reshape_1/shape?
time_distributed_24/Reshape_1Reshape-time_distributed_24/dense_24/BiasAdd:output:0,time_distributed_24/Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_24/Reshape_1?
#time_distributed_24/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2%
#time_distributed_24/Reshape_2/shape?
time_distributed_24/Reshape_2Reshapedropout_12/Identity:output:0,time_distributed_24/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape_2?
!time_distributed_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!time_distributed_25/Reshape/shape?
time_distributed_25/ReshapeReshape&time_distributed_24/Reshape_1:output:0*time_distributed_25/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape?
2time_distributed_25/dense_25/MatMul/ReadVariableOpReadVariableOp;time_distributed_25_dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype024
2time_distributed_25/dense_25/MatMul/ReadVariableOp?
#time_distributed_25/dense_25/MatMulMatMul$time_distributed_25/Reshape:output:0:time_distributed_25/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#time_distributed_25/dense_25/MatMul?
3time_distributed_25/dense_25/BiasAdd/ReadVariableOpReadVariableOp<time_distributed_25_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3time_distributed_25/dense_25/BiasAdd/ReadVariableOp?
$time_distributed_25/dense_25/BiasAddBiasAdd-time_distributed_25/dense_25/MatMul:product:0;time_distributed_25/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$time_distributed_25/dense_25/BiasAdd?
#time_distributed_25/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2%
#time_distributed_25/Reshape_1/shape?
time_distributed_25/Reshape_1Reshape-time_distributed_25/dense_25/BiasAdd:output:0,time_distributed_25/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
time_distributed_25/Reshape_1?
#time_distributed_25/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2%
#time_distributed_25/Reshape_2/shape?
time_distributed_25/Reshape_2Reshape&time_distributed_24/Reshape_1:output:0,time_distributed_25/Reshape_2/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape_2?
IdentityIdentity&time_distributed_25/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv1d_24/BiasAdd/ReadVariableOp-^conv1d_24/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_25/BiasAdd/ReadVariableOp-^conv1d_25/conv1d/ExpandDims_1/ReadVariableOp"^gru_12/gru_cell_12/ReadVariableOp$^gru_12/gru_cell_12/ReadVariableOp_1$^gru_12/gru_cell_12/ReadVariableOp_2$^gru_12/gru_cell_12/ReadVariableOp_3$^gru_12/gru_cell_12/ReadVariableOp_4$^gru_12/gru_cell_12/ReadVariableOp_5$^gru_12/gru_cell_12/ReadVariableOp_6$^gru_12/gru_cell_12/ReadVariableOp_7$^gru_12/gru_cell_12/ReadVariableOp_8^gru_12/while4^time_distributed_24/dense_24/BiasAdd/ReadVariableOp3^time_distributed_24/dense_24/MatMul/ReadVariableOp4^time_distributed_25/dense_25/BiasAdd/ReadVariableOp3^time_distributed_25/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2D
 conv1d_24/BiasAdd/ReadVariableOp conv1d_24/BiasAdd/ReadVariableOp2\
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_25/BiasAdd/ReadVariableOp conv1d_25/BiasAdd/ReadVariableOp2\
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp2F
!gru_12/gru_cell_12/ReadVariableOp!gru_12/gru_cell_12/ReadVariableOp2J
#gru_12/gru_cell_12/ReadVariableOp_1#gru_12/gru_cell_12/ReadVariableOp_12J
#gru_12/gru_cell_12/ReadVariableOp_2#gru_12/gru_cell_12/ReadVariableOp_22J
#gru_12/gru_cell_12/ReadVariableOp_3#gru_12/gru_cell_12/ReadVariableOp_32J
#gru_12/gru_cell_12/ReadVariableOp_4#gru_12/gru_cell_12/ReadVariableOp_42J
#gru_12/gru_cell_12/ReadVariableOp_5#gru_12/gru_cell_12/ReadVariableOp_52J
#gru_12/gru_cell_12/ReadVariableOp_6#gru_12/gru_cell_12/ReadVariableOp_62J
#gru_12/gru_cell_12/ReadVariableOp_7#gru_12/gru_cell_12/ReadVariableOp_72J
#gru_12/gru_cell_12/ReadVariableOp_8#gru_12/gru_cell_12/ReadVariableOp_82
gru_12/whilegru_12/while2j
3time_distributed_24/dense_24/BiasAdd/ReadVariableOp3time_distributed_24/dense_24/BiasAdd/ReadVariableOp2h
2time_distributed_24/dense_24/MatMul/ReadVariableOp2time_distributed_24/dense_24/MatMul/ReadVariableOp2j
3time_distributed_25/dense_25/BiasAdd/ReadVariableOp3time_distributed_25/dense_25/BiasAdd/ReadVariableOp2h
2time_distributed_25/dense_25/MatMul/ReadVariableOp2time_distributed_25/dense_25/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_12_layer_call_fn_292667

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
GPU2*0J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2900862
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
?

?
D__inference_dense_25_layer_call_and_return_conditional_losses_289626

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
'__inference_gru_12_layer_call_fn_291616
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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2892312
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
?
?
4__inference_time_distributed_25_layer_call_fn_292831

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2901812
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
h
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_291594

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
?
h
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_291586

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
?
?
while_cond_290373
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_290373___redundant_placeholder04
0while_while_cond_290373___redundant_placeholder14
0while_while_cond_290373___redundant_placeholder24
0while_while_cond_290373___redundant_placeholder3
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
?a
?
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_288909

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
?
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_292677

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
?
?
while_cond_289934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_289934___redundant_placeholder04
0while_while_cond_289934___redundant_placeholder14
0while_while_cond_289934___redundant_placeholder24
0while_while_cond_289934___redundant_placeholder3
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
?

?
,__inference_gru_cell_12_layer_call_fn_292915

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
GPU2*0J 8? *P
fKRI
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_2889092
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
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292746

inputs9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource: 
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOpD
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
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulReshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/BiasAddq
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
	Reshape_1Reshapedense_24/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_290240

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
?	
?
gru_12_while_cond_290995*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2,
(gru_12_while_less_gru_12_strided_slice_1B
>gru_12_while_gru_12_while_cond_290995___redundant_placeholder0B
>gru_12_while_gru_12_while_cond_290995___redundant_placeholder1B
>gru_12_while_gru_12_while_cond_290995___redundant_placeholder2B
>gru_12_while_gru_12_while_cond_290995___redundant_placeholder3
gru_12_while_identity
?
gru_12/while/LessLessgru_12_while_placeholder(gru_12_while_less_gru_12_strided_slice_1*
T0*
_output_shapes
: 2
gru_12/while/Lessr
gru_12/while/IdentityIdentitygru_12/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_12/while/Identity"7
gru_12_while_identitygru_12/while/Identity:output:0*(
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
4__inference_time_distributed_24_layer_call_fn_292725

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2902132
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
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292795

inputs9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource: 
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOpo
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
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulReshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense_24/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292852

inputs9
'dense_25_matmul_readvariableop_resource: 6
(dense_25_biasadd_readvariableop_resource:
identity??dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOpD
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
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMulReshape:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAddq
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
	Reshape_1Reshapedense_25/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
??
?	
while_body_289935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_12_readvariableop_resource_0:	?H;
-while_gru_cell_12_readvariableop_3_resource_0:H?
-while_gru_cell_12_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_12_readvariableop_resource:	?H9
+while_gru_cell_12_readvariableop_3_resource:H=
+while_gru_cell_12_readvariableop_6_resource:H?? while/gru_cell_12/ReadVariableOp?"while/gru_cell_12/ReadVariableOp_1?"while/gru_cell_12/ReadVariableOp_2?"while/gru_cell_12/ReadVariableOp_3?"while/gru_cell_12/ReadVariableOp_4?"while/gru_cell_12/ReadVariableOp_5?"while/gru_cell_12/ReadVariableOp_6?"while/gru_cell_12/ReadVariableOp_7?"while/gru_cell_12/ReadVariableOp_8?
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
 while/gru_cell_12/ReadVariableOpReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02"
 while/gru_cell_12/ReadVariableOp?
%while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/gru_cell_12/strided_slice/stack?
'while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice/stack_1?
'while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell_12/strided_slice/stack_2?
while/gru_cell_12/strided_sliceStridedSlice(while/gru_cell_12/ReadVariableOp:value:0.while/gru_cell_12/strided_slice/stack:output:00while/gru_cell_12/strided_slice/stack_1:output:00while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2!
while/gru_cell_12/strided_slice?
while/gru_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0(while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul?
"while/gru_cell_12/ReadVariableOp_1ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_1?
'while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_1/stack?
)while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_1/stack_1?
)while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_1/stack_2?
!while/gru_cell_12/strided_slice_1StridedSlice*while/gru_cell_12/ReadVariableOp_1:value:00while/gru_cell_12/strided_slice_1/stack:output:02while/gru_cell_12/strided_slice_1/stack_1:output:02while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_1?
while/gru_cell_12/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_1?
"while/gru_cell_12/ReadVariableOp_2ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_2?
'while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_2/stack?
)while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_2/stack_1?
)while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_2/stack_2?
!while/gru_cell_12/strided_slice_2StridedSlice*while/gru_cell_12/ReadVariableOp_2:value:00while/gru_cell_12/strided_slice_2/stack:output:02while/gru_cell_12/strided_slice_2/stack_1:output:02while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_2?
while/gru_cell_12/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_2?
"while/gru_cell_12/ReadVariableOp_3ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_3?
'while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell_12/strided_slice_3/stack?
)while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_1?
)while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_2?
!while/gru_cell_12/strided_slice_3StridedSlice*while/gru_cell_12/ReadVariableOp_3:value:00while/gru_cell_12/strided_slice_3/stack:output:02while/gru_cell_12/strided_slice_3/stack_1:output:02while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2#
!while/gru_cell_12/strided_slice_3?
while/gru_cell_12/BiasAddBiasAdd"while/gru_cell_12/MatMul:product:0*while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd?
"while/gru_cell_12/ReadVariableOp_4ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_4?
'while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell_12/strided_slice_4/stack?
)while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02+
)while/gru_cell_12/strided_slice_4/stack_1?
)while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_4/stack_2?
!while/gru_cell_12/strided_slice_4StridedSlice*while/gru_cell_12/ReadVariableOp_4:value:00while/gru_cell_12/strided_slice_4/stack:output:02while/gru_cell_12/strided_slice_4/stack_1:output:02while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!while/gru_cell_12/strided_slice_4?
while/gru_cell_12/BiasAdd_1BiasAdd$while/gru_cell_12/MatMul_1:product:0*while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_1?
"while/gru_cell_12/ReadVariableOp_5ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_5?
'while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02)
'while/gru_cell_12/strided_slice_5/stack?
)while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_12/strided_slice_5/stack_1?
)while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_5/stack_2?
!while/gru_cell_12/strided_slice_5StridedSlice*while/gru_cell_12/ReadVariableOp_5:value:00while/gru_cell_12/strided_slice_5/stack:output:02while/gru_cell_12/strided_slice_5/stack_1:output:02while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2#
!while/gru_cell_12/strided_slice_5?
while/gru_cell_12/BiasAdd_2BiasAdd$while/gru_cell_12/MatMul_2:product:0*while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_2?
"while/gru_cell_12/ReadVariableOp_6ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_6?
'while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell_12/strided_slice_6/stack?
)while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/gru_cell_12/strided_slice_6/stack_1?
)while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_6/stack_2?
!while/gru_cell_12/strided_slice_6StridedSlice*while/gru_cell_12/ReadVariableOp_6:value:00while/gru_cell_12/strided_slice_6/stack:output:02while/gru_cell_12/strided_slice_6/stack_1:output:02while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_6?
while/gru_cell_12/MatMul_3MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_3?
"while/gru_cell_12/ReadVariableOp_7ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_7?
'while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_7/stack?
)while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_7/stack_1?
)while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_7/stack_2?
!while/gru_cell_12/strided_slice_7StridedSlice*while/gru_cell_12/ReadVariableOp_7:value:00while/gru_cell_12/strided_slice_7/stack:output:02while/gru_cell_12/strided_slice_7/stack_1:output:02while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_7?
while/gru_cell_12/MatMul_4MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_4?
while/gru_cell_12/addAddV2"while/gru_cell_12/BiasAdd:output:0$while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/addw
while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const{
while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_1?
while/gru_cell_12/MulMulwhile/gru_cell_12/add:z:0 while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul?
while/gru_cell_12/Add_1AddV2while/gru_cell_12/Mul:z:0"while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_1?
)while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/gru_cell_12/clip_by_value/Minimum/y?
'while/gru_cell_12/clip_by_value/MinimumMinimumwhile/gru_cell_12/Add_1:z:02while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2)
'while/gru_cell_12/clip_by_value/Minimum?
!while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/gru_cell_12/clip_by_value/y?
while/gru_cell_12/clip_by_valueMaximum+while/gru_cell_12/clip_by_value/Minimum:z:0*while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2!
while/gru_cell_12/clip_by_value?
while/gru_cell_12/add_2AddV2$while/gru_cell_12/BiasAdd_1:output:0$while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_2{
while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const_2{
while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_3?
while/gru_cell_12/Mul_1Mulwhile/gru_cell_12/add_2:z:0"while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul_1?
while/gru_cell_12/Add_3AddV2while/gru_cell_12/Mul_1:z:0"while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_3?
+while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+while/gru_cell_12/clip_by_value_1/Minimum/y?
)while/gru_cell_12/clip_by_value_1/MinimumMinimumwhile/gru_cell_12/Add_3:z:04while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)while/gru_cell_12/clip_by_value_1/Minimum?
#while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#while/gru_cell_12/clip_by_value_1/y?
!while/gru_cell_12/clip_by_value_1Maximum-while/gru_cell_12/clip_by_value_1/Minimum:z:0,while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2#
!while/gru_cell_12/clip_by_value_1?
while/gru_cell_12/mul_2Mul%while/gru_cell_12/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_2?
"while/gru_cell_12/ReadVariableOp_8ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_8?
'while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_8/stack?
)while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_8/stack_1?
)while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_8/stack_2?
!while/gru_cell_12/strided_slice_8StridedSlice*while/gru_cell_12/ReadVariableOp_8:value:00while/gru_cell_12/strided_slice_8/stack:output:02while/gru_cell_12/strided_slice_8/stack_1:output:02while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_8?
while/gru_cell_12/MatMul_5MatMulwhile/gru_cell_12/mul_2:z:0*while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_5?
while/gru_cell_12/add_4AddV2$while/gru_cell_12/BiasAdd_2:output:0$while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_4?
while/gru_cell_12/ReluReluwhile/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Relu?
while/gru_cell_12/mul_3Mul#while/gru_cell_12/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_3w
while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_12/sub/x?
while/gru_cell_12/subSub while/gru_cell_12/sub/x:output:0#while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/sub?
while/gru_cell_12/mul_4Mulwhile/gru_cell_12/sub:z:0$while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_4?
while/gru_cell_12/add_5AddV2while/gru_cell_12/mul_3:z:0while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_12/add_5:z:0*
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
while/Identity_4Identitywhile/gru_cell_12/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp!^while/gru_cell_12/ReadVariableOp#^while/gru_cell_12/ReadVariableOp_1#^while/gru_cell_12/ReadVariableOp_2#^while/gru_cell_12/ReadVariableOp_3#^while/gru_cell_12/ReadVariableOp_4#^while/gru_cell_12/ReadVariableOp_5#^while/gru_cell_12/ReadVariableOp_6#^while/gru_cell_12/ReadVariableOp_7#^while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"\
+while_gru_cell_12_readvariableop_3_resource-while_gru_cell_12_readvariableop_3_resource_0"\
+while_gru_cell_12_readvariableop_6_resource-while_gru_cell_12_readvariableop_6_resource_0"X
)while_gru_cell_12_readvariableop_resource+while_gru_cell_12_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2D
 while/gru_cell_12/ReadVariableOp while/gru_cell_12/ReadVariableOp2H
"while/gru_cell_12/ReadVariableOp_1"while/gru_cell_12/ReadVariableOp_12H
"while/gru_cell_12/ReadVariableOp_2"while/gru_cell_12/ReadVariableOp_22H
"while/gru_cell_12/ReadVariableOp_3"while/gru_cell_12/ReadVariableOp_32H
"while/gru_cell_12/ReadVariableOp_4"while/gru_cell_12/ReadVariableOp_42H
"while/gru_cell_12/ReadVariableOp_5"while/gru_cell_12/ReadVariableOp_52H
"while/gru_cell_12/ReadVariableOp_6"while/gru_cell_12/ReadVariableOp_62H
"while/gru_cell_12/ReadVariableOp_7"while/gru_cell_12/ReadVariableOp_72H
"while/gru_cell_12/ReadVariableOp_8"while/gru_cell_12/ReadVariableOp_8: 
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
?
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_290086

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
?
?
while_cond_292011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_292011___redundant_placeholder04
0while_while_cond_292011___redundant_placeholder14
0while_while_cond_292011___redundant_placeholder24
0while_while_cond_292011___redundant_placeholder3
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
?

?
,__inference_gru_cell_12_layer_call_fn_292929

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
GPU2*0J 8? *P
fKRI
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_2891012
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
?
?
*__inference_conv1d_24_layer_call_fn_291490

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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2897642
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
?	
?
gru_12_while_cond_291310*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2,
(gru_12_while_less_gru_12_strided_slice_1B
>gru_12_while_gru_12_while_cond_291310___redundant_placeholder0B
>gru_12_while_gru_12_while_cond_291310___redundant_placeholder1B
>gru_12_while_gru_12_while_cond_291310___redundant_placeholder2B
>gru_12_while_gru_12_while_cond_291310___redundant_placeholder3
gru_12_while_identity
?
gru_12/while/LessLessgru_12_while_placeholder(gru_12_while_less_gru_12_strided_slice_1*
T0*
_output_shapes
: 2
gru_12/while/Lessr
gru_12/while/IdentityIdentitygru_12/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_12/while/Identity"7
gru_12_while_identitygru_12/while/Identity:output:0*(
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
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_290101

inputs9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource: 
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOpo
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
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulReshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape_1/shape?
	Reshape_1Reshapedense_24/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?a
?
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_293018

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
ɼ
?
"__inference__traced_restore_293436
file_prefix7
!assignvariableop_conv1d_24_kernel: /
!assignvariableop_1_conv1d_24_bias: 9
#assignvariableop_2_conv1d_25_kernel: @/
!assignvariableop_3_conv1d_25_bias:@'
assignvariableop_4_nadam_iter:	 )
assignvariableop_5_nadam_beta_1: )
assignvariableop_6_nadam_beta_2: (
assignvariableop_7_nadam_decay: 0
&assignvariableop_8_nadam_learning_rate: 1
'assignvariableop_9_nadam_momentum_cache: @
-assignvariableop_10_gru_12_gru_cell_12_kernel:	?HI
7assignvariableop_11_gru_12_gru_cell_12_recurrent_kernel:H9
+assignvariableop_12_gru_12_gru_cell_12_bias:H@
.assignvariableop_13_time_distributed_24_kernel: :
,assignvariableop_14_time_distributed_24_bias: @
.assignvariableop_15_time_distributed_25_kernel: :
,assignvariableop_16_time_distributed_25_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: B
,assignvariableop_21_nadam_conv1d_24_kernel_m: 8
*assignvariableop_22_nadam_conv1d_24_bias_m: B
,assignvariableop_23_nadam_conv1d_25_kernel_m: @8
*assignvariableop_24_nadam_conv1d_25_bias_m:@H
5assignvariableop_25_nadam_gru_12_gru_cell_12_kernel_m:	?HQ
?assignvariableop_26_nadam_gru_12_gru_cell_12_recurrent_kernel_m:HA
3assignvariableop_27_nadam_gru_12_gru_cell_12_bias_m:HH
6assignvariableop_28_nadam_time_distributed_24_kernel_m: B
4assignvariableop_29_nadam_time_distributed_24_bias_m: H
6assignvariableop_30_nadam_time_distributed_25_kernel_m: B
4assignvariableop_31_nadam_time_distributed_25_bias_m:B
,assignvariableop_32_nadam_conv1d_24_kernel_v: 8
*assignvariableop_33_nadam_conv1d_24_bias_v: B
,assignvariableop_34_nadam_conv1d_25_kernel_v: @8
*assignvariableop_35_nadam_conv1d_25_bias_v:@H
5assignvariableop_36_nadam_gru_12_gru_cell_12_kernel_v:	?HQ
?assignvariableop_37_nadam_gru_12_gru_cell_12_recurrent_kernel_v:HA
3assignvariableop_38_nadam_gru_12_gru_cell_12_bias_v:HH
6assignvariableop_39_nadam_time_distributed_24_kernel_v: B
4assignvariableop_40_nadam_time_distributed_24_bias_v: H
6assignvariableop_41_nadam_time_distributed_25_kernel_v: B
4assignvariableop_42_nadam_time_distributed_25_bias_v:
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_24_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_25_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_25_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_10AssignVariableOp-assignvariableop_10_gru_12_gru_cell_12_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp7assignvariableop_11_gru_12_gru_cell_12_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_gru_12_gru_cell_12_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp.assignvariableop_13_time_distributed_24_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp,assignvariableop_14_time_distributed_24_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_time_distributed_25_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_time_distributed_25_biasIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_21AssignVariableOp,assignvariableop_21_nadam_conv1d_24_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_nadam_conv1d_24_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_nadam_conv1d_25_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_nadam_conv1d_25_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp5assignvariableop_25_nadam_gru_12_gru_cell_12_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp?assignvariableop_26_nadam_gru_12_gru_cell_12_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp3assignvariableop_27_nadam_gru_12_gru_cell_12_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_nadam_time_distributed_24_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_nadam_time_distributed_24_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_nadam_time_distributed_25_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp4assignvariableop_31_nadam_time_distributed_25_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_nadam_conv1d_24_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_nadam_conv1d_24_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp,assignvariableop_34_nadam_conv1d_25_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_nadam_conv1d_25_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp5assignvariableop_36_nadam_gru_12_gru_cell_12_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp?assignvariableop_37_nadam_gru_12_gru_cell_12_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_nadam_gru_12_gru_cell_12_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_nadam_time_distributed_24_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp4assignvariableop_40_nadam_time_distributed_24_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp6assignvariableop_41_nadam_time_distributed_25_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_nadam_time_distributed_25_bias_vIdentity_42:output:0"/device:CPU:0*
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
?a
?
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_289101

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
'__inference_gru_12_layer_call_fn_291638

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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2905122
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
?

?
D__inference_dense_24_layer_call_and_return_conditional_losses_293126

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
??
?	
while_body_291756
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_12_readvariableop_resource_0:	?H;
-while_gru_cell_12_readvariableop_3_resource_0:H?
-while_gru_cell_12_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_12_readvariableop_resource:	?H9
+while_gru_cell_12_readvariableop_3_resource:H=
+while_gru_cell_12_readvariableop_6_resource:H?? while/gru_cell_12/ReadVariableOp?"while/gru_cell_12/ReadVariableOp_1?"while/gru_cell_12/ReadVariableOp_2?"while/gru_cell_12/ReadVariableOp_3?"while/gru_cell_12/ReadVariableOp_4?"while/gru_cell_12/ReadVariableOp_5?"while/gru_cell_12/ReadVariableOp_6?"while/gru_cell_12/ReadVariableOp_7?"while/gru_cell_12/ReadVariableOp_8?
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
 while/gru_cell_12/ReadVariableOpReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02"
 while/gru_cell_12/ReadVariableOp?
%while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/gru_cell_12/strided_slice/stack?
'while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice/stack_1?
'while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell_12/strided_slice/stack_2?
while/gru_cell_12/strided_sliceStridedSlice(while/gru_cell_12/ReadVariableOp:value:0.while/gru_cell_12/strided_slice/stack:output:00while/gru_cell_12/strided_slice/stack_1:output:00while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2!
while/gru_cell_12/strided_slice?
while/gru_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0(while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul?
"while/gru_cell_12/ReadVariableOp_1ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_1?
'while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_1/stack?
)while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_1/stack_1?
)while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_1/stack_2?
!while/gru_cell_12/strided_slice_1StridedSlice*while/gru_cell_12/ReadVariableOp_1:value:00while/gru_cell_12/strided_slice_1/stack:output:02while/gru_cell_12/strided_slice_1/stack_1:output:02while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_1?
while/gru_cell_12/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_1?
"while/gru_cell_12/ReadVariableOp_2ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_2?
'while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_2/stack?
)while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_2/stack_1?
)while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_2/stack_2?
!while/gru_cell_12/strided_slice_2StridedSlice*while/gru_cell_12/ReadVariableOp_2:value:00while/gru_cell_12/strided_slice_2/stack:output:02while/gru_cell_12/strided_slice_2/stack_1:output:02while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_2?
while/gru_cell_12/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_2?
"while/gru_cell_12/ReadVariableOp_3ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_3?
'while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell_12/strided_slice_3/stack?
)while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_1?
)while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_2?
!while/gru_cell_12/strided_slice_3StridedSlice*while/gru_cell_12/ReadVariableOp_3:value:00while/gru_cell_12/strided_slice_3/stack:output:02while/gru_cell_12/strided_slice_3/stack_1:output:02while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2#
!while/gru_cell_12/strided_slice_3?
while/gru_cell_12/BiasAddBiasAdd"while/gru_cell_12/MatMul:product:0*while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd?
"while/gru_cell_12/ReadVariableOp_4ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_4?
'while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell_12/strided_slice_4/stack?
)while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02+
)while/gru_cell_12/strided_slice_4/stack_1?
)while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_4/stack_2?
!while/gru_cell_12/strided_slice_4StridedSlice*while/gru_cell_12/ReadVariableOp_4:value:00while/gru_cell_12/strided_slice_4/stack:output:02while/gru_cell_12/strided_slice_4/stack_1:output:02while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!while/gru_cell_12/strided_slice_4?
while/gru_cell_12/BiasAdd_1BiasAdd$while/gru_cell_12/MatMul_1:product:0*while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_1?
"while/gru_cell_12/ReadVariableOp_5ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_5?
'while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02)
'while/gru_cell_12/strided_slice_5/stack?
)while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_12/strided_slice_5/stack_1?
)while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_5/stack_2?
!while/gru_cell_12/strided_slice_5StridedSlice*while/gru_cell_12/ReadVariableOp_5:value:00while/gru_cell_12/strided_slice_5/stack:output:02while/gru_cell_12/strided_slice_5/stack_1:output:02while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2#
!while/gru_cell_12/strided_slice_5?
while/gru_cell_12/BiasAdd_2BiasAdd$while/gru_cell_12/MatMul_2:product:0*while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_2?
"while/gru_cell_12/ReadVariableOp_6ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_6?
'while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell_12/strided_slice_6/stack?
)while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/gru_cell_12/strided_slice_6/stack_1?
)while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_6/stack_2?
!while/gru_cell_12/strided_slice_6StridedSlice*while/gru_cell_12/ReadVariableOp_6:value:00while/gru_cell_12/strided_slice_6/stack:output:02while/gru_cell_12/strided_slice_6/stack_1:output:02while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_6?
while/gru_cell_12/MatMul_3MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_3?
"while/gru_cell_12/ReadVariableOp_7ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_7?
'while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_7/stack?
)while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_7/stack_1?
)while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_7/stack_2?
!while/gru_cell_12/strided_slice_7StridedSlice*while/gru_cell_12/ReadVariableOp_7:value:00while/gru_cell_12/strided_slice_7/stack:output:02while/gru_cell_12/strided_slice_7/stack_1:output:02while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_7?
while/gru_cell_12/MatMul_4MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_4?
while/gru_cell_12/addAddV2"while/gru_cell_12/BiasAdd:output:0$while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/addw
while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const{
while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_1?
while/gru_cell_12/MulMulwhile/gru_cell_12/add:z:0 while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul?
while/gru_cell_12/Add_1AddV2while/gru_cell_12/Mul:z:0"while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_1?
)while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/gru_cell_12/clip_by_value/Minimum/y?
'while/gru_cell_12/clip_by_value/MinimumMinimumwhile/gru_cell_12/Add_1:z:02while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2)
'while/gru_cell_12/clip_by_value/Minimum?
!while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/gru_cell_12/clip_by_value/y?
while/gru_cell_12/clip_by_valueMaximum+while/gru_cell_12/clip_by_value/Minimum:z:0*while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2!
while/gru_cell_12/clip_by_value?
while/gru_cell_12/add_2AddV2$while/gru_cell_12/BiasAdd_1:output:0$while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_2{
while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const_2{
while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_3?
while/gru_cell_12/Mul_1Mulwhile/gru_cell_12/add_2:z:0"while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul_1?
while/gru_cell_12/Add_3AddV2while/gru_cell_12/Mul_1:z:0"while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_3?
+while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+while/gru_cell_12/clip_by_value_1/Minimum/y?
)while/gru_cell_12/clip_by_value_1/MinimumMinimumwhile/gru_cell_12/Add_3:z:04while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)while/gru_cell_12/clip_by_value_1/Minimum?
#while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#while/gru_cell_12/clip_by_value_1/y?
!while/gru_cell_12/clip_by_value_1Maximum-while/gru_cell_12/clip_by_value_1/Minimum:z:0,while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2#
!while/gru_cell_12/clip_by_value_1?
while/gru_cell_12/mul_2Mul%while/gru_cell_12/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_2?
"while/gru_cell_12/ReadVariableOp_8ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_8?
'while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_8/stack?
)while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_8/stack_1?
)while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_8/stack_2?
!while/gru_cell_12/strided_slice_8StridedSlice*while/gru_cell_12/ReadVariableOp_8:value:00while/gru_cell_12/strided_slice_8/stack:output:02while/gru_cell_12/strided_slice_8/stack_1:output:02while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_8?
while/gru_cell_12/MatMul_5MatMulwhile/gru_cell_12/mul_2:z:0*while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_5?
while/gru_cell_12/add_4AddV2$while/gru_cell_12/BiasAdd_2:output:0$while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_4?
while/gru_cell_12/ReluReluwhile/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Relu?
while/gru_cell_12/mul_3Mul#while/gru_cell_12/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_3w
while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_12/sub/x?
while/gru_cell_12/subSub while/gru_cell_12/sub/x:output:0#while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/sub?
while/gru_cell_12/mul_4Mulwhile/gru_cell_12/sub:z:0$while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_4?
while/gru_cell_12/add_5AddV2while/gru_cell_12/mul_3:z:0while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_12/add_5:z:0*
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
while/Identity_4Identitywhile/gru_cell_12/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp!^while/gru_cell_12/ReadVariableOp#^while/gru_cell_12/ReadVariableOp_1#^while/gru_cell_12/ReadVariableOp_2#^while/gru_cell_12/ReadVariableOp_3#^while/gru_cell_12/ReadVariableOp_4#^while/gru_cell_12/ReadVariableOp_5#^while/gru_cell_12/ReadVariableOp_6#^while/gru_cell_12/ReadVariableOp_7#^while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"\
+while_gru_cell_12_readvariableop_3_resource-while_gru_cell_12_readvariableop_3_resource_0"\
+while_gru_cell_12_readvariableop_6_resource-while_gru_cell_12_readvariableop_6_resource_0"X
)while_gru_cell_12_readvariableop_resource+while_gru_cell_12_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2D
 while/gru_cell_12/ReadVariableOp while/gru_cell_12/ReadVariableOp2H
"while/gru_cell_12/ReadVariableOp_1"while/gru_cell_12/ReadVariableOp_12H
"while/gru_cell_12/ReadVariableOp_2"while/gru_cell_12/ReadVariableOp_22H
"while/gru_cell_12/ReadVariableOp_3"while/gru_cell_12/ReadVariableOp_32H
"while/gru_cell_12/ReadVariableOp_4"while/gru_cell_12/ReadVariableOp_42H
"while/gru_cell_12/ReadVariableOp_5"while/gru_cell_12/ReadVariableOp_52H
"while/gru_cell_12/ReadVariableOp_6"while/gru_cell_12/ReadVariableOp_62H
"while/gru_cell_12/ReadVariableOp_7"while/gru_cell_12/ReadVariableOp_72H
"while/gru_cell_12/ReadVariableOp_8"while/gru_cell_12/ReadVariableOp_8: 
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
h
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_288741

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
??
?
&sequential_12_gru_12_while_body_288566F
Bsequential_12_gru_12_while_sequential_12_gru_12_while_loop_counterL
Hsequential_12_gru_12_while_sequential_12_gru_12_while_maximum_iterations*
&sequential_12_gru_12_while_placeholder,
(sequential_12_gru_12_while_placeholder_1,
(sequential_12_gru_12_while_placeholder_2E
Asequential_12_gru_12_while_sequential_12_gru_12_strided_slice_1_0?
}sequential_12_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_12_gru_12_tensorarrayunstack_tensorlistfromtensor_0S
@sequential_12_gru_12_while_gru_cell_12_readvariableop_resource_0:	?HP
Bsequential_12_gru_12_while_gru_cell_12_readvariableop_3_resource_0:HT
Bsequential_12_gru_12_while_gru_cell_12_readvariableop_6_resource_0:H'
#sequential_12_gru_12_while_identity)
%sequential_12_gru_12_while_identity_1)
%sequential_12_gru_12_while_identity_2)
%sequential_12_gru_12_while_identity_3)
%sequential_12_gru_12_while_identity_4C
?sequential_12_gru_12_while_sequential_12_gru_12_strided_slice_1
{sequential_12_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_12_gru_12_tensorarrayunstack_tensorlistfromtensorQ
>sequential_12_gru_12_while_gru_cell_12_readvariableop_resource:	?HN
@sequential_12_gru_12_while_gru_cell_12_readvariableop_3_resource:HR
@sequential_12_gru_12_while_gru_cell_12_readvariableop_6_resource:H??5sequential_12/gru_12/while/gru_cell_12/ReadVariableOp?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_1?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_2?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_3?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_4?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_5?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_6?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_7?7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_8?
Lsequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2N
Lsequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_12_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_12_gru_12_tensorarrayunstack_tensorlistfromtensor_0&sequential_12_gru_12_while_placeholderUsequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02@
>sequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItem?
5sequential_12/gru_12/while/gru_cell_12/ReadVariableOpReadVariableOp@sequential_12_gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype027
5sequential_12/gru_12/while/gru_cell_12/ReadVariableOp?
:sequential_12/gru_12/while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_12/gru_12/while/gru_cell_12/strided_slice/stack?
<sequential_12/gru_12/while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice/stack_1?
<sequential_12/gru_12/while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice/stack_2?
4sequential_12/gru_12/while/gru_cell_12/strided_sliceStridedSlice=sequential_12/gru_12/while/gru_cell_12/ReadVariableOp:value:0Csequential_12/gru_12/while/gru_cell_12/strided_slice/stack:output:0Esequential_12/gru_12/while/gru_cell_12/strided_slice/stack_1:output:0Esequential_12/gru_12/while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask26
4sequential_12/gru_12/while/gru_cell_12/strided_slice?
-sequential_12/gru_12/while/gru_cell_12/MatMulMatMulEsequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential_12/gru_12/while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2/
-sequential_12/gru_12/while/gru_cell_12/MatMul?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_1ReadVariableOp@sequential_12_gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_1?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_1StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_1:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_1?
/sequential_12/gru_12/while/gru_cell_12/MatMul_1MatMulEsequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItem:item:0?sequential_12/gru_12/while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????21
/sequential_12/gru_12/while/gru_cell_12/MatMul_1?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_2ReadVariableOp@sequential_12_gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_2?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_2StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_2:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_2?
/sequential_12/gru_12/while/gru_cell_12/MatMul_2MatMulEsequential_12/gru_12/while/TensorArrayV2Read/TensorListGetItem:item:0?sequential_12/gru_12/while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????21
/sequential_12/gru_12/while/gru_cell_12/MatMul_2?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_3ReadVariableOpBsequential_12_gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_3?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_3StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_3:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_3?
.sequential_12/gru_12/while/gru_cell_12/BiasAddBiasAdd7sequential_12/gru_12/while/gru_cell_12/MatMul:product:0?sequential_12/gru_12/while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????20
.sequential_12/gru_12/while/gru_cell_12/BiasAdd?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_4ReadVariableOpBsequential_12_gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_4?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_4StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_4:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_4?
0sequential_12/gru_12/while/gru_cell_12/BiasAdd_1BiasAdd9sequential_12/gru_12/while/gru_cell_12/MatMul_1:product:0?sequential_12/gru_12/while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????22
0sequential_12/gru_12/while/gru_cell_12/BiasAdd_1?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_5ReadVariableOpBsequential_12_gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_5?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_5StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_5:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_5?
0sequential_12/gru_12/while/gru_cell_12/BiasAdd_2BiasAdd9sequential_12/gru_12/while/gru_cell_12/MatMul_2:product:0?sequential_12/gru_12/while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????22
0sequential_12/gru_12/while/gru_cell_12/BiasAdd_2?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_6ReadVariableOpBsequential_12_gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_6?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_6StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_6:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_6?
/sequential_12/gru_12/while/gru_cell_12/MatMul_3MatMul(sequential_12_gru_12_while_placeholder_2?sequential_12/gru_12/while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????21
/sequential_12/gru_12/while/gru_cell_12/MatMul_3?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_7ReadVariableOpBsequential_12_gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_7?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_7StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_7:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_7?
/sequential_12/gru_12/while/gru_cell_12/MatMul_4MatMul(sequential_12_gru_12_while_placeholder_2?sequential_12/gru_12/while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????21
/sequential_12/gru_12/while/gru_cell_12/MatMul_4?
*sequential_12/gru_12/while/gru_cell_12/addAddV27sequential_12/gru_12/while/gru_cell_12/BiasAdd:output:09sequential_12/gru_12/while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2,
*sequential_12/gru_12/while/gru_cell_12/add?
,sequential_12/gru_12/while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,sequential_12/gru_12/while/gru_cell_12/Const?
.sequential_12/gru_12/while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_12/gru_12/while/gru_cell_12/Const_1?
*sequential_12/gru_12/while/gru_cell_12/MulMul.sequential_12/gru_12/while/gru_cell_12/add:z:05sequential_12/gru_12/while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2,
*sequential_12/gru_12/while/gru_cell_12/Mul?
,sequential_12/gru_12/while/gru_cell_12/Add_1AddV2.sequential_12/gru_12/while/gru_cell_12/Mul:z:07sequential_12/gru_12/while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/Add_1?
>sequential_12/gru_12/while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2@
>sequential_12/gru_12/while/gru_cell_12/clip_by_value/Minimum/y?
<sequential_12/gru_12/while/gru_cell_12/clip_by_value/MinimumMinimum0sequential_12/gru_12/while/gru_cell_12/Add_1:z:0Gsequential_12/gru_12/while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2>
<sequential_12/gru_12/while/gru_cell_12/clip_by_value/Minimum?
6sequential_12/gru_12/while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_12/gru_12/while/gru_cell_12/clip_by_value/y?
4sequential_12/gru_12/while/gru_cell_12/clip_by_valueMaximum@sequential_12/gru_12/while/gru_cell_12/clip_by_value/Minimum:z:0?sequential_12/gru_12/while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????26
4sequential_12/gru_12/while/gru_cell_12/clip_by_value?
,sequential_12/gru_12/while/gru_cell_12/add_2AddV29sequential_12/gru_12/while/gru_cell_12/BiasAdd_1:output:09sequential_12/gru_12/while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/add_2?
.sequential_12/gru_12/while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>20
.sequential_12/gru_12/while/gru_cell_12/Const_2?
.sequential_12/gru_12/while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_12/gru_12/while/gru_cell_12/Const_3?
,sequential_12/gru_12/while/gru_cell_12/Mul_1Mul0sequential_12/gru_12/while/gru_cell_12/add_2:z:07sequential_12/gru_12/while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/Mul_1?
,sequential_12/gru_12/while/gru_cell_12/Add_3AddV20sequential_12/gru_12/while/gru_cell_12/Mul_1:z:07sequential_12/gru_12/while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/Add_3?
@sequential_12/gru_12/while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@sequential_12/gru_12/while/gru_cell_12/clip_by_value_1/Minimum/y?
>sequential_12/gru_12/while/gru_cell_12/clip_by_value_1/MinimumMinimum0sequential_12/gru_12/while/gru_cell_12/Add_3:z:0Isequential_12/gru_12/while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2@
>sequential_12/gru_12/while/gru_cell_12/clip_by_value_1/Minimum?
8sequential_12/gru_12/while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8sequential_12/gru_12/while/gru_cell_12/clip_by_value_1/y?
6sequential_12/gru_12/while/gru_cell_12/clip_by_value_1MaximumBsequential_12/gru_12/while/gru_cell_12/clip_by_value_1/Minimum:z:0Asequential_12/gru_12/while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????28
6sequential_12/gru_12/while/gru_cell_12/clip_by_value_1?
,sequential_12/gru_12/while/gru_cell_12/mul_2Mul:sequential_12/gru_12/while/gru_cell_12/clip_by_value_1:z:0(sequential_12_gru_12_while_placeholder_2*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/mul_2?
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_8ReadVariableOpBsequential_12_gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype029
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_8?
<sequential_12/gru_12/while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2>
<sequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack_1?
>sequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack_2?
6sequential_12/gru_12/while/gru_cell_12/strided_slice_8StridedSlice?sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_8:value:0Esequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack_1:output:0Gsequential_12/gru_12/while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask28
6sequential_12/gru_12/while/gru_cell_12/strided_slice_8?
/sequential_12/gru_12/while/gru_cell_12/MatMul_5MatMul0sequential_12/gru_12/while/gru_cell_12/mul_2:z:0?sequential_12/gru_12/while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????21
/sequential_12/gru_12/while/gru_cell_12/MatMul_5?
,sequential_12/gru_12/while/gru_cell_12/add_4AddV29sequential_12/gru_12/while/gru_cell_12/BiasAdd_2:output:09sequential_12/gru_12/while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/add_4?
+sequential_12/gru_12/while/gru_cell_12/ReluRelu0sequential_12/gru_12/while/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2-
+sequential_12/gru_12/while/gru_cell_12/Relu?
,sequential_12/gru_12/while/gru_cell_12/mul_3Mul8sequential_12/gru_12/while/gru_cell_12/clip_by_value:z:0(sequential_12_gru_12_while_placeholder_2*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/mul_3?
,sequential_12/gru_12/while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,sequential_12/gru_12/while/gru_cell_12/sub/x?
*sequential_12/gru_12/while/gru_cell_12/subSub5sequential_12/gru_12/while/gru_cell_12/sub/x:output:08sequential_12/gru_12/while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2,
*sequential_12/gru_12/while/gru_cell_12/sub?
,sequential_12/gru_12/while/gru_cell_12/mul_4Mul.sequential_12/gru_12/while/gru_cell_12/sub:z:09sequential_12/gru_12/while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/mul_4?
,sequential_12/gru_12/while/gru_cell_12/add_5AddV20sequential_12/gru_12/while/gru_cell_12/mul_3:z:00sequential_12/gru_12/while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2.
,sequential_12/gru_12/while/gru_cell_12/add_5?
?sequential_12/gru_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_12_gru_12_while_placeholder_1&sequential_12_gru_12_while_placeholder0sequential_12/gru_12/while/gru_cell_12/add_5:z:0*
_output_shapes
: *
element_dtype02A
?sequential_12/gru_12/while/TensorArrayV2Write/TensorListSetItem?
 sequential_12/gru_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_12/gru_12/while/add/y?
sequential_12/gru_12/while/addAddV2&sequential_12_gru_12_while_placeholder)sequential_12/gru_12/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_12/gru_12/while/add?
"sequential_12/gru_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_12/gru_12/while/add_1/y?
 sequential_12/gru_12/while/add_1AddV2Bsequential_12_gru_12_while_sequential_12_gru_12_while_loop_counter+sequential_12/gru_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/gru_12/while/add_1?
#sequential_12/gru_12/while/IdentityIdentity$sequential_12/gru_12/while/add_1:z:0 ^sequential_12/gru_12/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_12/gru_12/while/Identity?
%sequential_12/gru_12/while/Identity_1IdentityHsequential_12_gru_12_while_sequential_12_gru_12_while_maximum_iterations ^sequential_12/gru_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_12/gru_12/while/Identity_1?
%sequential_12/gru_12/while/Identity_2Identity"sequential_12/gru_12/while/add:z:0 ^sequential_12/gru_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_12/gru_12/while/Identity_2?
%sequential_12/gru_12/while/Identity_3IdentityOsequential_12/gru_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_12/gru_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_12/gru_12/while/Identity_3?
%sequential_12/gru_12/while/Identity_4Identity0sequential_12/gru_12/while/gru_cell_12/add_5:z:0 ^sequential_12/gru_12/while/NoOp*
T0*'
_output_shapes
:?????????2'
%sequential_12/gru_12/while/Identity_4?
sequential_12/gru_12/while/NoOpNoOp6^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp8^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_18^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_28^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_38^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_48^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_58^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_68^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_78^sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_12/gru_12/while/NoOp"?
@sequential_12_gru_12_while_gru_cell_12_readvariableop_3_resourceBsequential_12_gru_12_while_gru_cell_12_readvariableop_3_resource_0"?
@sequential_12_gru_12_while_gru_cell_12_readvariableop_6_resourceBsequential_12_gru_12_while_gru_cell_12_readvariableop_6_resource_0"?
>sequential_12_gru_12_while_gru_cell_12_readvariableop_resource@sequential_12_gru_12_while_gru_cell_12_readvariableop_resource_0"S
#sequential_12_gru_12_while_identity,sequential_12/gru_12/while/Identity:output:0"W
%sequential_12_gru_12_while_identity_1.sequential_12/gru_12/while/Identity_1:output:0"W
%sequential_12_gru_12_while_identity_2.sequential_12/gru_12/while/Identity_2:output:0"W
%sequential_12_gru_12_while_identity_3.sequential_12/gru_12/while/Identity_3:output:0"W
%sequential_12_gru_12_while_identity_4.sequential_12/gru_12/while/Identity_4:output:0"?
?sequential_12_gru_12_while_sequential_12_gru_12_strided_slice_1Asequential_12_gru_12_while_sequential_12_gru_12_strided_slice_1_0"?
{sequential_12_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_12_gru_12_tensorarrayunstack_tensorlistfromtensor}sequential_12_gru_12_while_tensorarrayv2read_tensorlistgetitem_sequential_12_gru_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2n
5sequential_12/gru_12/while/gru_cell_12/ReadVariableOp5sequential_12/gru_12/while/gru_cell_12/ReadVariableOp2r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_17sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_12r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_27sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_22r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_37sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_32r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_47sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_42r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_57sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_52r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_67sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_62r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_77sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_72r
7sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_87sequential_12/gru_12/while/gru_cell_12/ReadVariableOp_8: 
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
?
M
1__inference_max_pooling1d_12_layer_call_fn_291536

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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_2887412
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
?
?
while_cond_292523
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_292523___redundant_placeholder04
0while_while_cond_292523___redundant_placeholder14
0while_while_cond_292523___redundant_placeholder24
0while_while_cond_292523___redundant_placeholder3
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
?
?
E__inference_conv1d_25_layer_call_and_return_conditional_losses_291531

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
while_cond_289167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_289167___redundant_placeholder04
0while_while_cond_289167___redundant_placeholder14
0while_while_cond_289167___redundant_placeholder24
0while_while_cond_289167___redundant_placeholder3
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
?
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_292689

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
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292901

inputs9
'dense_25_matmul_readvariableop_resource: 6
(dense_25_biasadd_readvariableop_resource:
identity??dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOpo
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
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMulReshape:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_25/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
4__inference_time_distributed_24_layer_call_fn_292716

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2901012
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
?
?
)__inference_dense_25_layer_call_fn_293135

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
GPU2*0J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_2896262
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
??
?
B__inference_gru_12_layer_call_and_return_conditional_losses_292662

inputs6
#gru_cell_12_readvariableop_resource:	?H3
%gru_cell_12_readvariableop_3_resource:H7
%gru_cell_12_readvariableop_6_resource:H
identity??gru_cell_12/ReadVariableOp?gru_cell_12/ReadVariableOp_1?gru_cell_12/ReadVariableOp_2?gru_cell_12/ReadVariableOp_3?gru_cell_12/ReadVariableOp_4?gru_cell_12/ReadVariableOp_5?gru_cell_12/ReadVariableOp_6?gru_cell_12/ReadVariableOp_7?gru_cell_12/ReadVariableOp_8?whileD
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
gru_cell_12/ReadVariableOpReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp?
gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
gru_cell_12/strided_slice/stack?
!gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice/stack_1?
!gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell_12/strided_slice/stack_2?
gru_cell_12/strided_sliceStridedSlice"gru_cell_12/ReadVariableOp:value:0(gru_cell_12/strided_slice/stack:output:0*gru_cell_12/strided_slice/stack_1:output:0*gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice?
gru_cell_12/MatMulMatMulstrided_slice_2:output:0"gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul?
gru_cell_12/ReadVariableOp_1ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_1?
!gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_1/stack?
#gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_1/stack_1?
#gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_1/stack_2?
gru_cell_12/strided_slice_1StridedSlice$gru_cell_12/ReadVariableOp_1:value:0*gru_cell_12/strided_slice_1/stack:output:0,gru_cell_12/strided_slice_1/stack_1:output:0,gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_1?
gru_cell_12/MatMul_1MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_1?
gru_cell_12/ReadVariableOp_2ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_2?
!gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_2/stack?
#gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_2/stack_1?
#gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_2/stack_2?
gru_cell_12/strided_slice_2StridedSlice$gru_cell_12/ReadVariableOp_2:value:0*gru_cell_12/strided_slice_2/stack:output:0,gru_cell_12/strided_slice_2/stack_1:output:0,gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_2?
gru_cell_12/MatMul_2MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_2?
gru_cell_12/ReadVariableOp_3ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_3?
!gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell_12/strided_slice_3/stack?
#gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_1?
#gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_2?
gru_cell_12/strided_slice_3StridedSlice$gru_cell_12/ReadVariableOp_3:value:0*gru_cell_12/strided_slice_3/stack:output:0,gru_cell_12/strided_slice_3/stack_1:output:0,gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell_12/strided_slice_3?
gru_cell_12/BiasAddBiasAddgru_cell_12/MatMul:product:0$gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd?
gru_cell_12/ReadVariableOp_4ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_4?
!gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell_12/strided_slice_4/stack?
#gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02%
#gru_cell_12/strided_slice_4/stack_1?
#gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_4/stack_2?
gru_cell_12/strided_slice_4StridedSlice$gru_cell_12/ReadVariableOp_4:value:0*gru_cell_12/strided_slice_4/stack:output:0,gru_cell_12/strided_slice_4/stack_1:output:0,gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell_12/strided_slice_4?
gru_cell_12/BiasAdd_1BiasAddgru_cell_12/MatMul_1:product:0$gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_1?
gru_cell_12/ReadVariableOp_5ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_5?
!gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02#
!gru_cell_12/strided_slice_5/stack?
#gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_12/strided_slice_5/stack_1?
#gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_5/stack_2?
gru_cell_12/strided_slice_5StridedSlice$gru_cell_12/ReadVariableOp_5:value:0*gru_cell_12/strided_slice_5/stack:output:0,gru_cell_12/strided_slice_5/stack_1:output:0,gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell_12/strided_slice_5?
gru_cell_12/BiasAdd_2BiasAddgru_cell_12/MatMul_2:product:0$gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_2?
gru_cell_12/ReadVariableOp_6ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_6?
!gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell_12/strided_slice_6/stack?
#gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#gru_cell_12/strided_slice_6/stack_1?
#gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_6/stack_2?
gru_cell_12/strided_slice_6StridedSlice$gru_cell_12/ReadVariableOp_6:value:0*gru_cell_12/strided_slice_6/stack:output:0,gru_cell_12/strided_slice_6/stack_1:output:0,gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_6?
gru_cell_12/MatMul_3MatMulzeros:output:0$gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_3?
gru_cell_12/ReadVariableOp_7ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_7?
!gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_7/stack?
#gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_7/stack_1?
#gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_7/stack_2?
gru_cell_12/strided_slice_7StridedSlice$gru_cell_12/ReadVariableOp_7:value:0*gru_cell_12/strided_slice_7/stack:output:0,gru_cell_12/strided_slice_7/stack_1:output:0,gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_7?
gru_cell_12/MatMul_4MatMulzeros:output:0$gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_4?
gru_cell_12/addAddV2gru_cell_12/BiasAdd:output:0gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/addk
gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Consto
gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_1?
gru_cell_12/MulMulgru_cell_12/add:z:0gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul?
gru_cell_12/Add_1AddV2gru_cell_12/Mul:z:0gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_1?
#gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#gru_cell_12/clip_by_value/Minimum/y?
!gru_cell_12/clip_by_value/MinimumMinimumgru_cell_12/Add_1:z:0,gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2#
!gru_cell_12/clip_by_value/Minimum
gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value/y?
gru_cell_12/clip_by_valueMaximum%gru_cell_12/clip_by_value/Minimum:z:0$gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value?
gru_cell_12/add_2AddV2gru_cell_12/BiasAdd_1:output:0gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_2o
gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Const_2o
gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_3?
gru_cell_12/Mul_1Mulgru_cell_12/add_2:z:0gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul_1?
gru_cell_12/Add_3AddV2gru_cell_12/Mul_1:z:0gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_3?
%gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gru_cell_12/clip_by_value_1/Minimum/y?
#gru_cell_12/clip_by_value_1/MinimumMinimumgru_cell_12/Add_3:z:0.gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2%
#gru_cell_12/clip_by_value_1/Minimum?
gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value_1/y?
gru_cell_12/clip_by_value_1Maximum'gru_cell_12/clip_by_value_1/Minimum:z:0&gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value_1?
gru_cell_12/mul_2Mulgru_cell_12/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_2?
gru_cell_12/ReadVariableOp_8ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_8?
!gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_8/stack?
#gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_8/stack_1?
#gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_8/stack_2?
gru_cell_12/strided_slice_8StridedSlice$gru_cell_12/ReadVariableOp_8:value:0*gru_cell_12/strided_slice_8/stack:output:0,gru_cell_12/strided_slice_8/stack_1:output:0,gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_8?
gru_cell_12/MatMul_5MatMulgru_cell_12/mul_2:z:0$gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_5?
gru_cell_12/add_4AddV2gru_cell_12/BiasAdd_2:output:0gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_4u
gru_cell_12/ReluRelugru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Relu?
gru_cell_12/mul_3Mulgru_cell_12/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_3k
gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_12/sub/x?
gru_cell_12/subSubgru_cell_12/sub/x:output:0gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/sub?
gru_cell_12/mul_4Mulgru_cell_12/sub:z:0gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_4?
gru_cell_12/add_5AddV2gru_cell_12/mul_3:z:0gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_5?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_12_readvariableop_resource%gru_cell_12_readvariableop_3_resource%gru_cell_12_readvariableop_6_resource*
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
bodyR
while_body_292524*
condR
while_cond_292523*8
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
NoOpNoOp^gru_cell_12/ReadVariableOp^gru_cell_12/ReadVariableOp_1^gru_cell_12/ReadVariableOp_2^gru_cell_12/ReadVariableOp_3^gru_cell_12/ReadVariableOp_4^gru_cell_12/ReadVariableOp_5^gru_cell_12/ReadVariableOp_6^gru_cell_12/ReadVariableOp_7^gru_cell_12/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
gru_cell_12/ReadVariableOpgru_cell_12/ReadVariableOp2<
gru_cell_12/ReadVariableOp_1gru_cell_12/ReadVariableOp_12<
gru_cell_12/ReadVariableOp_2gru_cell_12/ReadVariableOp_22<
gru_cell_12/ReadVariableOp_3gru_cell_12/ReadVariableOp_32<
gru_cell_12/ReadVariableOp_4gru_cell_12/ReadVariableOp_42<
gru_cell_12/ReadVariableOp_5gru_cell_12/ReadVariableOp_52<
gru_cell_12/ReadVariableOp_6gru_cell_12/ReadVariableOp_62<
gru_cell_12/ReadVariableOp_7gru_cell_12/ReadVariableOp_72<
gru_cell_12/ReadVariableOp_8gru_cell_12/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292887

inputs9
'dense_25_matmul_readvariableop_resource: 6
(dense_25_biasadd_readvariableop_resource:
identity??dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOpo
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
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMulReshape:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAddw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapedense_25/BiasAdd:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1q
IdentityIdentityReshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
4__inference_time_distributed_24_layer_call_fn_292698

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2894982
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
?
&sequential_12_gru_12_while_cond_288565F
Bsequential_12_gru_12_while_sequential_12_gru_12_while_loop_counterL
Hsequential_12_gru_12_while_sequential_12_gru_12_while_maximum_iterations*
&sequential_12_gru_12_while_placeholder,
(sequential_12_gru_12_while_placeholder_1,
(sequential_12_gru_12_while_placeholder_2H
Dsequential_12_gru_12_while_less_sequential_12_gru_12_strided_slice_1^
Zsequential_12_gru_12_while_sequential_12_gru_12_while_cond_288565___redundant_placeholder0^
Zsequential_12_gru_12_while_sequential_12_gru_12_while_cond_288565___redundant_placeholder1^
Zsequential_12_gru_12_while_sequential_12_gru_12_while_cond_288565___redundant_placeholder2^
Zsequential_12_gru_12_while_sequential_12_gru_12_while_cond_288565___redundant_placeholder3'
#sequential_12_gru_12_while_identity
?
sequential_12/gru_12/while/LessLess&sequential_12_gru_12_while_placeholderDsequential_12_gru_12_while_less_sequential_12_gru_12_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_12/gru_12/while/Less?
#sequential_12/gru_12/while/IdentityIdentity#sequential_12/gru_12/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_12/gru_12/while/Identity"S
#sequential_12_gru_12_while_identity,sequential_12/gru_12/while/Identity:output:0*(
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
?2
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_290716
conv1d_24_input&
conv1d_24_290680: 
conv1d_24_290682: &
conv1d_25_290685: @
conv1d_25_290687:@ 
gru_12_290693:	?H
gru_12_290695:H
gru_12_290697:H,
time_distributed_24_290701: (
time_distributed_24_290703: ,
time_distributed_25_290708: (
time_distributed_25_290710:
identity??!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?gru_12/StatefulPartitionedCall?+time_distributed_24/StatefulPartitionedCall?+time_distributed_25/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCallconv1d_24_inputconv1d_24_290680conv1d_24_290682*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2897642#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_290685conv1d_25_290687*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2897862#
!conv1d_25/StatefulPartitionedCall?
 max_pooling1d_12/PartitionedCallPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_2897992"
 max_pooling1d_12/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling1d_12/PartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_2898072
flatten_12/PartitionedCall?
 repeat_vector_12/PartitionedCallPartitionedCall#flatten_12/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_2898162"
 repeat_vector_12/PartitionedCall?
gru_12/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_12/PartitionedCall:output:0gru_12_290693gru_12_290695gru_12_290697*
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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2900732 
gru_12/StatefulPartitionedCall?
dropout_12/PartitionedCallPartitionedCall'gru_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2900862
dropout_12/PartitionedCall?
+time_distributed_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0time_distributed_24_290701time_distributed_24_290703*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2901012-
+time_distributed_24/StatefulPartitionedCall?
!time_distributed_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!time_distributed_24/Reshape/shape?
time_distributed_24/ReshapeReshape#dropout_12/PartitionedCall:output:0*time_distributed_24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape?
+time_distributed_25/StatefulPartitionedCallStatefulPartitionedCall4time_distributed_24/StatefulPartitionedCall:output:0time_distributed_25_290708time_distributed_25_290710*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2901222-
+time_distributed_25/StatefulPartitionedCall?
!time_distributed_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!time_distributed_25/Reshape/shape?
time_distributed_25/ReshapeReshape4time_distributed_24/StatefulPartitionedCall:output:0*time_distributed_25/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape?
IdentityIdentity4time_distributed_25/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall^gru_12/StatefulPartitionedCall,^time_distributed_24/StatefulPartitionedCall,^time_distributed_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2Z
+time_distributed_24/StatefulPartitionedCall+time_distributed_24/StatefulPartitionedCall2Z
+time_distributed_25/StatefulPartitionedCall+time_distributed_25/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameconv1d_24_input
?
?
*__inference_conv1d_25_layer_call_fn_291515

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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2897862
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
?
G
+__inference_flatten_12_layer_call_fn_291562

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
GPU2*0J 8? *O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_2898072
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
?
?
.__inference_sequential_12_layer_call_fn_290817

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
GPU2*0J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_2901312
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
?2
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_290131

inputs&
conv1d_24_289765: 
conv1d_24_289767: &
conv1d_25_289787: @
conv1d_25_289789:@ 
gru_12_290074:	?H
gru_12_290076:H
gru_12_290078:H,
time_distributed_24_290102: (
time_distributed_24_290104: ,
time_distributed_25_290123: (
time_distributed_25_290125:
identity??!conv1d_24/StatefulPartitionedCall?!conv1d_25/StatefulPartitionedCall?gru_12/StatefulPartitionedCall?+time_distributed_24/StatefulPartitionedCall?+time_distributed_25/StatefulPartitionedCall?
!conv1d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_24_289765conv1d_24_289767*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_24_layer_call_and_return_conditional_losses_2897642#
!conv1d_24/StatefulPartitionedCall?
!conv1d_25/StatefulPartitionedCallStatefulPartitionedCall*conv1d_24/StatefulPartitionedCall:output:0conv1d_25_289787conv1d_25_289789*
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
GPU2*0J 8? *N
fIRG
E__inference_conv1d_25_layer_call_and_return_conditional_losses_2897862#
!conv1d_25/StatefulPartitionedCall?
 max_pooling1d_12/PartitionedCallPartitionedCall*conv1d_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_2897992"
 max_pooling1d_12/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling1d_12/PartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_2898072
flatten_12/PartitionedCall?
 repeat_vector_12/PartitionedCallPartitionedCall#flatten_12/PartitionedCall:output:0*
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
GPU2*0J 8? *U
fPRN
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_2898162"
 repeat_vector_12/PartitionedCall?
gru_12/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_12/PartitionedCall:output:0gru_12_290074gru_12_290076gru_12_290078*
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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2900732 
gru_12/StatefulPartitionedCall?
dropout_12/PartitionedCallPartitionedCall'gru_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2900862
dropout_12/PartitionedCall?
+time_distributed_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0time_distributed_24_290102time_distributed_24_290104*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2901012-
+time_distributed_24/StatefulPartitionedCall?
!time_distributed_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!time_distributed_24/Reshape/shape?
time_distributed_24/ReshapeReshape#dropout_12/PartitionedCall:output:0*time_distributed_24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape?
+time_distributed_25/StatefulPartitionedCallStatefulPartitionedCall4time_distributed_24/StatefulPartitionedCall:output:0time_distributed_25_290123time_distributed_25_290125*
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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2901222-
+time_distributed_25/StatefulPartitionedCall?
!time_distributed_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!time_distributed_25/Reshape/shape?
time_distributed_25/ReshapeReshape4time_distributed_24/StatefulPartitionedCall:output:0*time_distributed_25/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape?
IdentityIdentity4time_distributed_25/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp"^conv1d_24/StatefulPartitionedCall"^conv1d_25/StatefulPartitionedCall^gru_12/StatefulPartitionedCall,^time_distributed_24/StatefulPartitionedCall,^time_distributed_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2F
!conv1d_24/StatefulPartitionedCall!conv1d_24/StatefulPartitionedCall2F
!conv1d_25/StatefulPartitionedCall!conv1d_25/StatefulPartitionedCall2@
gru_12/StatefulPartitionedCallgru_12/StatefulPartitionedCall2Z
+time_distributed_24/StatefulPartitionedCall+time_distributed_24/StatefulPartitionedCall2Z
+time_distributed_25/StatefulPartitionedCall+time_distributed_25/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_25_layer_call_and_return_conditional_losses_289786

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
??
?
gru_12_while_body_290996*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2)
%gru_12_while_gru_12_strided_slice_1_0e
agru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0E
2gru_12_while_gru_cell_12_readvariableop_resource_0:	?HB
4gru_12_while_gru_cell_12_readvariableop_3_resource_0:HF
4gru_12_while_gru_cell_12_readvariableop_6_resource_0:H
gru_12_while_identity
gru_12_while_identity_1
gru_12_while_identity_2
gru_12_while_identity_3
gru_12_while_identity_4'
#gru_12_while_gru_12_strided_slice_1c
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensorC
0gru_12_while_gru_cell_12_readvariableop_resource:	?H@
2gru_12_while_gru_cell_12_readvariableop_3_resource:HD
2gru_12_while_gru_cell_12_readvariableop_6_resource:H??'gru_12/while/gru_cell_12/ReadVariableOp?)gru_12/while/gru_cell_12/ReadVariableOp_1?)gru_12/while/gru_cell_12/ReadVariableOp_2?)gru_12/while/gru_cell_12/ReadVariableOp_3?)gru_12/while/gru_cell_12/ReadVariableOp_4?)gru_12/while/gru_cell_12/ReadVariableOp_5?)gru_12/while/gru_cell_12/ReadVariableOp_6?)gru_12/while/gru_cell_12/ReadVariableOp_7?)gru_12/while/gru_cell_12/ReadVariableOp_8?
>gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0gru_12_while_placeholderGgru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_12/while/TensorArrayV2Read/TensorListGetItem?
'gru_12/while/gru_cell_12/ReadVariableOpReadVariableOp2gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02)
'gru_12/while/gru_cell_12/ReadVariableOp?
,gru_12/while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,gru_12/while/gru_cell_12/strided_slice/stack?
.gru_12/while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.gru_12/while/gru_cell_12/strided_slice/stack_1?
.gru_12/while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_12/while/gru_cell_12/strided_slice/stack_2?
&gru_12/while/gru_cell_12/strided_sliceStridedSlice/gru_12/while/gru_cell_12/ReadVariableOp:value:05gru_12/while/gru_cell_12/strided_slice/stack:output:07gru_12/while/gru_cell_12/strided_slice/stack_1:output:07gru_12/while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2(
&gru_12/while/gru_cell_12/strided_slice?
gru_12/while/gru_cell_12/MatMulMatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_12/while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2!
gru_12/while/gru_cell_12/MatMul?
)gru_12/while/gru_cell_12/ReadVariableOp_1ReadVariableOp2gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_1?
.gru_12/while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.gru_12/while/gru_cell_12/strided_slice_1/stack?
0gru_12/while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   22
0gru_12/while/gru_cell_12/strided_slice_1/stack_1?
0gru_12/while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_1/stack_2?
(gru_12/while/gru_cell_12/strided_slice_1StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_1:value:07gru_12/while/gru_cell_12/strided_slice_1/stack:output:09gru_12/while/gru_cell_12/strided_slice_1/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_1?
!gru_12/while/gru_cell_12/MatMul_1MatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:01gru_12/while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_1?
)gru_12/while/gru_cell_12/ReadVariableOp_2ReadVariableOp2gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_2?
.gru_12/while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   20
.gru_12/while/gru_cell_12/strided_slice_2/stack?
0gru_12/while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0gru_12/while/gru_cell_12/strided_slice_2/stack_1?
0gru_12/while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_2/stack_2?
(gru_12/while/gru_cell_12/strided_slice_2StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_2:value:07gru_12/while/gru_cell_12/strided_slice_2/stack:output:09gru_12/while/gru_cell_12/strided_slice_2/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_2?
!gru_12/while/gru_cell_12/MatMul_2MatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:01gru_12/while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_2?
)gru_12/while/gru_cell_12/ReadVariableOp_3ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_3?
.gru_12/while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.gru_12/while/gru_cell_12/strided_slice_3/stack?
0gru_12/while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_3/stack_1?
0gru_12/while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_3/stack_2?
(gru_12/while/gru_cell_12/strided_slice_3StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_3:value:07gru_12/while/gru_cell_12/strided_slice_3/stack:output:09gru_12/while/gru_cell_12/strided_slice_3/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2*
(gru_12/while/gru_cell_12/strided_slice_3?
 gru_12/while/gru_cell_12/BiasAddBiasAdd)gru_12/while/gru_cell_12/MatMul:product:01gru_12/while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2"
 gru_12/while/gru_cell_12/BiasAdd?
)gru_12/while/gru_cell_12/ReadVariableOp_4ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_4?
.gru_12/while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.gru_12/while/gru_cell_12/strided_slice_4/stack?
0gru_12/while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:022
0gru_12/while/gru_cell_12/strided_slice_4/stack_1?
0gru_12/while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_4/stack_2?
(gru_12/while/gru_cell_12/strided_slice_4StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_4:value:07gru_12/while/gru_cell_12/strided_slice_4/stack:output:09gru_12/while/gru_cell_12/strided_slice_4/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2*
(gru_12/while/gru_cell_12/strided_slice_4?
"gru_12/while/gru_cell_12/BiasAdd_1BiasAdd+gru_12/while/gru_cell_12/MatMul_1:product:01gru_12/while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2$
"gru_12/while/gru_cell_12/BiasAdd_1?
)gru_12/while/gru_cell_12/ReadVariableOp_5ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_5?
.gru_12/while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:020
.gru_12/while/gru_cell_12/strided_slice_5/stack?
0gru_12/while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0gru_12/while/gru_cell_12/strided_slice_5/stack_1?
0gru_12/while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_5/stack_2?
(gru_12/while/gru_cell_12/strided_slice_5StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_5:value:07gru_12/while/gru_cell_12/strided_slice_5/stack:output:09gru_12/while/gru_cell_12/strided_slice_5/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_5?
"gru_12/while/gru_cell_12/BiasAdd_2BiasAdd+gru_12/while/gru_cell_12/MatMul_2:product:01gru_12/while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2$
"gru_12/while/gru_cell_12/BiasAdd_2?
)gru_12/while/gru_cell_12/ReadVariableOp_6ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_6?
.gru_12/while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.gru_12/while/gru_cell_12/strided_slice_6/stack?
0gru_12/while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0gru_12/while/gru_cell_12/strided_slice_6/stack_1?
0gru_12/while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_6/stack_2?
(gru_12/while/gru_cell_12/strided_slice_6StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_6:value:07gru_12/while/gru_cell_12/strided_slice_6/stack:output:09gru_12/while/gru_cell_12/strided_slice_6/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_6?
!gru_12/while/gru_cell_12/MatMul_3MatMulgru_12_while_placeholder_21gru_12/while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_3?
)gru_12/while/gru_cell_12/ReadVariableOp_7ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_7?
.gru_12/while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.gru_12/while/gru_cell_12/strided_slice_7/stack?
0gru_12/while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   22
0gru_12/while/gru_cell_12/strided_slice_7/stack_1?
0gru_12/while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_7/stack_2?
(gru_12/while/gru_cell_12/strided_slice_7StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_7:value:07gru_12/while/gru_cell_12/strided_slice_7/stack:output:09gru_12/while/gru_cell_12/strided_slice_7/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_7?
!gru_12/while/gru_cell_12/MatMul_4MatMulgru_12_while_placeholder_21gru_12/while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_4?
gru_12/while/gru_cell_12/addAddV2)gru_12/while/gru_cell_12/BiasAdd:output:0+gru_12/while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/add?
gru_12/while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
gru_12/while/gru_cell_12/Const?
 gru_12/while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 gru_12/while/gru_cell_12/Const_1?
gru_12/while/gru_cell_12/MulMul gru_12/while/gru_cell_12/add:z:0'gru_12/while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/Mul?
gru_12/while/gru_cell_12/Add_1AddV2 gru_12/while/gru_cell_12/Mul:z:0)gru_12/while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/Add_1?
0gru_12/while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0gru_12/while/gru_cell_12/clip_by_value/Minimum/y?
.gru_12/while/gru_cell_12/clip_by_value/MinimumMinimum"gru_12/while/gru_cell_12/Add_1:z:09gru_12/while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????20
.gru_12/while/gru_cell_12/clip_by_value/Minimum?
(gru_12/while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(gru_12/while/gru_cell_12/clip_by_value/y?
&gru_12/while/gru_cell_12/clip_by_valueMaximum2gru_12/while/gru_cell_12/clip_by_value/Minimum:z:01gru_12/while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2(
&gru_12/while/gru_cell_12/clip_by_value?
gru_12/while/gru_cell_12/add_2AddV2+gru_12/while/gru_cell_12/BiasAdd_1:output:0+gru_12/while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/add_2?
 gru_12/while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 gru_12/while/gru_cell_12/Const_2?
 gru_12/while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 gru_12/while/gru_cell_12/Const_3?
gru_12/while/gru_cell_12/Mul_1Mul"gru_12/while/gru_cell_12/add_2:z:0)gru_12/while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/Mul_1?
gru_12/while/gru_cell_12/Add_3AddV2"gru_12/while/gru_cell_12/Mul_1:z:0)gru_12/while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/Add_3?
2gru_12/while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2gru_12/while/gru_cell_12/clip_by_value_1/Minimum/y?
0gru_12/while/gru_cell_12/clip_by_value_1/MinimumMinimum"gru_12/while/gru_cell_12/Add_3:z:0;gru_12/while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0gru_12/while/gru_cell_12/clip_by_value_1/Minimum?
*gru_12/while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*gru_12/while/gru_cell_12/clip_by_value_1/y?
(gru_12/while/gru_cell_12/clip_by_value_1Maximum4gru_12/while/gru_cell_12/clip_by_value_1/Minimum:z:03gru_12/while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2*
(gru_12/while/gru_cell_12/clip_by_value_1?
gru_12/while/gru_cell_12/mul_2Mul,gru_12/while/gru_cell_12/clip_by_value_1:z:0gru_12_while_placeholder_2*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/mul_2?
)gru_12/while/gru_cell_12/ReadVariableOp_8ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_8?
.gru_12/while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   20
.gru_12/while/gru_cell_12/strided_slice_8/stack?
0gru_12/while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0gru_12/while/gru_cell_12/strided_slice_8/stack_1?
0gru_12/while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_8/stack_2?
(gru_12/while/gru_cell_12/strided_slice_8StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_8:value:07gru_12/while/gru_cell_12/strided_slice_8/stack:output:09gru_12/while/gru_cell_12/strided_slice_8/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_8?
!gru_12/while/gru_cell_12/MatMul_5MatMul"gru_12/while/gru_cell_12/mul_2:z:01gru_12/while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_5?
gru_12/while/gru_cell_12/add_4AddV2+gru_12/while/gru_cell_12/BiasAdd_2:output:0+gru_12/while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/add_4?
gru_12/while/gru_cell_12/ReluRelu"gru_12/while/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/Relu?
gru_12/while/gru_cell_12/mul_3Mul*gru_12/while/gru_cell_12/clip_by_value:z:0gru_12_while_placeholder_2*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/mul_3?
gru_12/while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_12/while/gru_cell_12/sub/x?
gru_12/while/gru_cell_12/subSub'gru_12/while/gru_cell_12/sub/x:output:0*gru_12/while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/sub?
gru_12/while/gru_cell_12/mul_4Mul gru_12/while/gru_cell_12/sub:z:0+gru_12/while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/mul_4?
gru_12/while/gru_cell_12/add_5AddV2"gru_12/while/gru_cell_12/mul_3:z:0"gru_12/while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/add_5?
1gru_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_12_while_placeholder_1gru_12_while_placeholder"gru_12/while/gru_cell_12/add_5:z:0*
_output_shapes
: *
element_dtype023
1gru_12/while/TensorArrayV2Write/TensorListSetItemj
gru_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_12/while/add/y?
gru_12/while/addAddV2gru_12_while_placeholdergru_12/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_12/while/addn
gru_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_12/while/add_1/y?
gru_12/while/add_1AddV2&gru_12_while_gru_12_while_loop_countergru_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_12/while/add_1?
gru_12/while/IdentityIdentitygru_12/while/add_1:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity?
gru_12/while/Identity_1Identity,gru_12_while_gru_12_while_maximum_iterations^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity_1?
gru_12/while/Identity_2Identitygru_12/while/add:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity_2?
gru_12/while/Identity_3IdentityAgru_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity_3?
gru_12/while/Identity_4Identity"gru_12/while/gru_cell_12/add_5:z:0^gru_12/while/NoOp*
T0*'
_output_shapes
:?????????2
gru_12/while/Identity_4?
gru_12/while/NoOpNoOp(^gru_12/while/gru_cell_12/ReadVariableOp*^gru_12/while/gru_cell_12/ReadVariableOp_1*^gru_12/while/gru_cell_12/ReadVariableOp_2*^gru_12/while/gru_cell_12/ReadVariableOp_3*^gru_12/while/gru_cell_12/ReadVariableOp_4*^gru_12/while/gru_cell_12/ReadVariableOp_5*^gru_12/while/gru_cell_12/ReadVariableOp_6*^gru_12/while/gru_cell_12/ReadVariableOp_7*^gru_12/while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
gru_12/while/NoOp"L
#gru_12_while_gru_12_strided_slice_1%gru_12_while_gru_12_strided_slice_1_0"j
2gru_12_while_gru_cell_12_readvariableop_3_resource4gru_12_while_gru_cell_12_readvariableop_3_resource_0"j
2gru_12_while_gru_cell_12_readvariableop_6_resource4gru_12_while_gru_cell_12_readvariableop_6_resource_0"f
0gru_12_while_gru_cell_12_readvariableop_resource2gru_12_while_gru_cell_12_readvariableop_resource_0"7
gru_12_while_identitygru_12/while/Identity:output:0";
gru_12_while_identity_1 gru_12/while/Identity_1:output:0";
gru_12_while_identity_2 gru_12/while/Identity_2:output:0";
gru_12_while_identity_3 gru_12/while/Identity_3:output:0";
gru_12_while_identity_4 gru_12/while/Identity_4:output:0"?
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensoragru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2R
'gru_12/while/gru_cell_12/ReadVariableOp'gru_12/while/gru_cell_12/ReadVariableOp2V
)gru_12/while/gru_cell_12/ReadVariableOp_1)gru_12/while/gru_cell_12/ReadVariableOp_12V
)gru_12/while/gru_cell_12/ReadVariableOp_2)gru_12/while/gru_cell_12/ReadVariableOp_22V
)gru_12/while/gru_cell_12/ReadVariableOp_3)gru_12/while/gru_cell_12/ReadVariableOp_32V
)gru_12/while/gru_cell_12/ReadVariableOp_4)gru_12/while/gru_cell_12/ReadVariableOp_42V
)gru_12/while/gru_cell_12/ReadVariableOp_5)gru_12/while/gru_cell_12/ReadVariableOp_52V
)gru_12/while/gru_cell_12/ReadVariableOp_6)gru_12/while/gru_cell_12/ReadVariableOp_62V
)gru_12/while/gru_cell_12/ReadVariableOp_7)gru_12/while/gru_cell_12/ReadVariableOp_72V
)gru_12/while/gru_cell_12/ReadVariableOp_8)gru_12/while/gru_cell_12/ReadVariableOp_8: 
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

?
D__inference_dense_24_layer_call_and_return_conditional_losses_289487

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
??
?	
while_body_292012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_12_readvariableop_resource_0:	?H;
-while_gru_cell_12_readvariableop_3_resource_0:H?
-while_gru_cell_12_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_12_readvariableop_resource:	?H9
+while_gru_cell_12_readvariableop_3_resource:H=
+while_gru_cell_12_readvariableop_6_resource:H?? while/gru_cell_12/ReadVariableOp?"while/gru_cell_12/ReadVariableOp_1?"while/gru_cell_12/ReadVariableOp_2?"while/gru_cell_12/ReadVariableOp_3?"while/gru_cell_12/ReadVariableOp_4?"while/gru_cell_12/ReadVariableOp_5?"while/gru_cell_12/ReadVariableOp_6?"while/gru_cell_12/ReadVariableOp_7?"while/gru_cell_12/ReadVariableOp_8?
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
 while/gru_cell_12/ReadVariableOpReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02"
 while/gru_cell_12/ReadVariableOp?
%while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/gru_cell_12/strided_slice/stack?
'while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice/stack_1?
'while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell_12/strided_slice/stack_2?
while/gru_cell_12/strided_sliceStridedSlice(while/gru_cell_12/ReadVariableOp:value:0.while/gru_cell_12/strided_slice/stack:output:00while/gru_cell_12/strided_slice/stack_1:output:00while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2!
while/gru_cell_12/strided_slice?
while/gru_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0(while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul?
"while/gru_cell_12/ReadVariableOp_1ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_1?
'while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_1/stack?
)while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_1/stack_1?
)while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_1/stack_2?
!while/gru_cell_12/strided_slice_1StridedSlice*while/gru_cell_12/ReadVariableOp_1:value:00while/gru_cell_12/strided_slice_1/stack:output:02while/gru_cell_12/strided_slice_1/stack_1:output:02while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_1?
while/gru_cell_12/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_1?
"while/gru_cell_12/ReadVariableOp_2ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_2?
'while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_2/stack?
)while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_2/stack_1?
)while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_2/stack_2?
!while/gru_cell_12/strided_slice_2StridedSlice*while/gru_cell_12/ReadVariableOp_2:value:00while/gru_cell_12/strided_slice_2/stack:output:02while/gru_cell_12/strided_slice_2/stack_1:output:02while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_2?
while/gru_cell_12/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_2?
"while/gru_cell_12/ReadVariableOp_3ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_3?
'while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell_12/strided_slice_3/stack?
)while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_1?
)while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_2?
!while/gru_cell_12/strided_slice_3StridedSlice*while/gru_cell_12/ReadVariableOp_3:value:00while/gru_cell_12/strided_slice_3/stack:output:02while/gru_cell_12/strided_slice_3/stack_1:output:02while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2#
!while/gru_cell_12/strided_slice_3?
while/gru_cell_12/BiasAddBiasAdd"while/gru_cell_12/MatMul:product:0*while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd?
"while/gru_cell_12/ReadVariableOp_4ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_4?
'while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell_12/strided_slice_4/stack?
)while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02+
)while/gru_cell_12/strided_slice_4/stack_1?
)while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_4/stack_2?
!while/gru_cell_12/strided_slice_4StridedSlice*while/gru_cell_12/ReadVariableOp_4:value:00while/gru_cell_12/strided_slice_4/stack:output:02while/gru_cell_12/strided_slice_4/stack_1:output:02while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!while/gru_cell_12/strided_slice_4?
while/gru_cell_12/BiasAdd_1BiasAdd$while/gru_cell_12/MatMul_1:product:0*while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_1?
"while/gru_cell_12/ReadVariableOp_5ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_5?
'while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02)
'while/gru_cell_12/strided_slice_5/stack?
)while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_12/strided_slice_5/stack_1?
)while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_5/stack_2?
!while/gru_cell_12/strided_slice_5StridedSlice*while/gru_cell_12/ReadVariableOp_5:value:00while/gru_cell_12/strided_slice_5/stack:output:02while/gru_cell_12/strided_slice_5/stack_1:output:02while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2#
!while/gru_cell_12/strided_slice_5?
while/gru_cell_12/BiasAdd_2BiasAdd$while/gru_cell_12/MatMul_2:product:0*while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_2?
"while/gru_cell_12/ReadVariableOp_6ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_6?
'while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell_12/strided_slice_6/stack?
)while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/gru_cell_12/strided_slice_6/stack_1?
)while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_6/stack_2?
!while/gru_cell_12/strided_slice_6StridedSlice*while/gru_cell_12/ReadVariableOp_6:value:00while/gru_cell_12/strided_slice_6/stack:output:02while/gru_cell_12/strided_slice_6/stack_1:output:02while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_6?
while/gru_cell_12/MatMul_3MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_3?
"while/gru_cell_12/ReadVariableOp_7ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_7?
'while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_7/stack?
)while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_7/stack_1?
)while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_7/stack_2?
!while/gru_cell_12/strided_slice_7StridedSlice*while/gru_cell_12/ReadVariableOp_7:value:00while/gru_cell_12/strided_slice_7/stack:output:02while/gru_cell_12/strided_slice_7/stack_1:output:02while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_7?
while/gru_cell_12/MatMul_4MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_4?
while/gru_cell_12/addAddV2"while/gru_cell_12/BiasAdd:output:0$while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/addw
while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const{
while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_1?
while/gru_cell_12/MulMulwhile/gru_cell_12/add:z:0 while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul?
while/gru_cell_12/Add_1AddV2while/gru_cell_12/Mul:z:0"while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_1?
)while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/gru_cell_12/clip_by_value/Minimum/y?
'while/gru_cell_12/clip_by_value/MinimumMinimumwhile/gru_cell_12/Add_1:z:02while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2)
'while/gru_cell_12/clip_by_value/Minimum?
!while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/gru_cell_12/clip_by_value/y?
while/gru_cell_12/clip_by_valueMaximum+while/gru_cell_12/clip_by_value/Minimum:z:0*while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2!
while/gru_cell_12/clip_by_value?
while/gru_cell_12/add_2AddV2$while/gru_cell_12/BiasAdd_1:output:0$while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_2{
while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const_2{
while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_3?
while/gru_cell_12/Mul_1Mulwhile/gru_cell_12/add_2:z:0"while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul_1?
while/gru_cell_12/Add_3AddV2while/gru_cell_12/Mul_1:z:0"while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_3?
+while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+while/gru_cell_12/clip_by_value_1/Minimum/y?
)while/gru_cell_12/clip_by_value_1/MinimumMinimumwhile/gru_cell_12/Add_3:z:04while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)while/gru_cell_12/clip_by_value_1/Minimum?
#while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#while/gru_cell_12/clip_by_value_1/y?
!while/gru_cell_12/clip_by_value_1Maximum-while/gru_cell_12/clip_by_value_1/Minimum:z:0,while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2#
!while/gru_cell_12/clip_by_value_1?
while/gru_cell_12/mul_2Mul%while/gru_cell_12/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_2?
"while/gru_cell_12/ReadVariableOp_8ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_8?
'while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_8/stack?
)while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_8/stack_1?
)while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_8/stack_2?
!while/gru_cell_12/strided_slice_8StridedSlice*while/gru_cell_12/ReadVariableOp_8:value:00while/gru_cell_12/strided_slice_8/stack:output:02while/gru_cell_12/strided_slice_8/stack_1:output:02while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_8?
while/gru_cell_12/MatMul_5MatMulwhile/gru_cell_12/mul_2:z:0*while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_5?
while/gru_cell_12/add_4AddV2$while/gru_cell_12/BiasAdd_2:output:0$while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_4?
while/gru_cell_12/ReluReluwhile/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Relu?
while/gru_cell_12/mul_3Mul#while/gru_cell_12/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_3w
while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_12/sub/x?
while/gru_cell_12/subSub while/gru_cell_12/sub/x:output:0#while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/sub?
while/gru_cell_12/mul_4Mulwhile/gru_cell_12/sub:z:0$while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_4?
while/gru_cell_12/add_5AddV2while/gru_cell_12/mul_3:z:0while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_12/add_5:z:0*
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
while/Identity_4Identitywhile/gru_cell_12/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp!^while/gru_cell_12/ReadVariableOp#^while/gru_cell_12/ReadVariableOp_1#^while/gru_cell_12/ReadVariableOp_2#^while/gru_cell_12/ReadVariableOp_3#^while/gru_cell_12/ReadVariableOp_4#^while/gru_cell_12/ReadVariableOp_5#^while/gru_cell_12/ReadVariableOp_6#^while/gru_cell_12/ReadVariableOp_7#^while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"\
+while_gru_cell_12_readvariableop_3_resource-while_gru_cell_12_readvariableop_3_resource_0"\
+while_gru_cell_12_readvariableop_6_resource-while_gru_cell_12_readvariableop_6_resource_0"X
)while_gru_cell_12_readvariableop_resource+while_gru_cell_12_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2D
 while/gru_cell_12/ReadVariableOp while/gru_cell_12/ReadVariableOp2H
"while/gru_cell_12/ReadVariableOp_1"while/gru_cell_12/ReadVariableOp_12H
"while/gru_cell_12/ReadVariableOp_2"while/gru_cell_12/ReadVariableOp_22H
"while/gru_cell_12/ReadVariableOp_3"while/gru_cell_12/ReadVariableOp_32H
"while/gru_cell_12/ReadVariableOp_4"while/gru_cell_12/ReadVariableOp_42H
"while/gru_cell_12/ReadVariableOp_5"while/gru_cell_12/ReadVariableOp_52H
"while/gru_cell_12/ReadVariableOp_6"while/gru_cell_12/ReadVariableOp_62H
"while/gru_cell_12/ReadVariableOp_7"while/gru_cell_12/ReadVariableOp_72H
"while/gru_cell_12/ReadVariableOp_8"while/gru_cell_12/ReadVariableOp_8: 
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
4__inference_time_distributed_24_layer_call_fn_292707

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_2895462
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
??
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_291481

inputsK
5conv1d_24_conv1d_expanddims_1_readvariableop_resource: 7
)conv1d_24_biasadd_readvariableop_resource: K
5conv1d_25_conv1d_expanddims_1_readvariableop_resource: @7
)conv1d_25_biasadd_readvariableop_resource:@=
*gru_12_gru_cell_12_readvariableop_resource:	?H:
,gru_12_gru_cell_12_readvariableop_3_resource:H>
,gru_12_gru_cell_12_readvariableop_6_resource:HM
;time_distributed_24_dense_24_matmul_readvariableop_resource: J
<time_distributed_24_dense_24_biasadd_readvariableop_resource: M
;time_distributed_25_dense_25_matmul_readvariableop_resource: J
<time_distributed_25_dense_25_biasadd_readvariableop_resource:
identity?? conv1d_24/BiasAdd/ReadVariableOp?,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp? conv1d_25/BiasAdd/ReadVariableOp?,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?!gru_12/gru_cell_12/ReadVariableOp?#gru_12/gru_cell_12/ReadVariableOp_1?#gru_12/gru_cell_12/ReadVariableOp_2?#gru_12/gru_cell_12/ReadVariableOp_3?#gru_12/gru_cell_12/ReadVariableOp_4?#gru_12/gru_cell_12/ReadVariableOp_5?#gru_12/gru_cell_12/ReadVariableOp_6?#gru_12/gru_cell_12/ReadVariableOp_7?#gru_12/gru_cell_12/ReadVariableOp_8?gru_12/while?3time_distributed_24/dense_24/BiasAdd/ReadVariableOp?2time_distributed_24/dense_24/MatMul/ReadVariableOp?3time_distributed_25/dense_25/BiasAdd/ReadVariableOp?2time_distributed_25/dense_25/MatMul/ReadVariableOp?
conv1d_24/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_24/conv1d/ExpandDims/dim?
conv1d_24/conv1d/ExpandDims
ExpandDimsinputs(conv1d_24/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_24/conv1d/ExpandDims?
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_24_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02.
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_24/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_24/conv1d/ExpandDims_1/dim?
conv1d_24/conv1d/ExpandDims_1
ExpandDims4conv1d_24/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_24/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_24/conv1d/ExpandDims_1?
conv1d_24/conv1dConv2D$conv1d_24/conv1d/ExpandDims:output:0&conv1d_24/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d_24/conv1d?
conv1d_24/conv1d/SqueezeSqueezeconv1d_24/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_24/conv1d/Squeeze?
 conv1d_24/BiasAdd/ReadVariableOpReadVariableOp)conv1d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_24/BiasAdd/ReadVariableOp?
conv1d_24/BiasAddBiasAdd!conv1d_24/conv1d/Squeeze:output:0(conv1d_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d_24/BiasAddz
conv1d_24/ReluReluconv1d_24/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
conv1d_24/Relu?
conv1d_25/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_25/conv1d/ExpandDims/dim?
conv1d_25/conv1d/ExpandDims
ExpandDimsconv1d_24/Relu:activations:0(conv1d_25/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d_25/conv1d/ExpandDims?
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_25_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02.
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_25/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_25/conv1d/ExpandDims_1/dim?
conv1d_25/conv1d/ExpandDims_1
ExpandDims4conv1d_25/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_25/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_25/conv1d/ExpandDims_1?
conv1d_25/conv1dConv2D$conv1d_25/conv1d/ExpandDims:output:0&conv1d_25/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_25/conv1d?
conv1d_25/conv1d/SqueezeSqueezeconv1d_25/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_25/conv1d/Squeeze?
 conv1d_25/BiasAdd/ReadVariableOpReadVariableOp)conv1d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_25/BiasAdd/ReadVariableOp?
conv1d_25/BiasAddBiasAdd!conv1d_25/conv1d/Squeeze:output:0(conv1d_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/BiasAddz
conv1d_25/ReluReluconv1d_25/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_25/Relu?
max_pooling1d_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_12/ExpandDims/dim?
max_pooling1d_12/ExpandDims
ExpandDimsconv1d_25/Relu:activations:0(max_pooling1d_12/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2
max_pooling1d_12/ExpandDims?
max_pooling1d_12/MaxPoolMaxPool$max_pooling1d_12/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_12/MaxPool?
max_pooling1d_12/SqueezeSqueeze!max_pooling1d_12/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2
max_pooling1d_12/Squeezeu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_12/Const?
flatten_12/ReshapeReshape!max_pooling1d_12/Squeeze:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
repeat_vector_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_12/ExpandDims/dim?
repeat_vector_12/ExpandDims
ExpandDimsflatten_12/Reshape:output:0(repeat_vector_12/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector_12/ExpandDims?
repeat_vector_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
repeat_vector_12/stack?
repeat_vector_12/TileTile$repeat_vector_12/ExpandDims:output:0repeat_vector_12/stack:output:0*
T0*,
_output_shapes
:??????????2
repeat_vector_12/Tilej
gru_12/ShapeShaperepeat_vector_12/Tile:output:0*
T0*
_output_shapes
:2
gru_12/Shape?
gru_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_12/strided_slice/stack?
gru_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_12/strided_slice/stack_1?
gru_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_12/strided_slice/stack_2?
gru_12/strided_sliceStridedSlicegru_12/Shape:output:0#gru_12/strided_slice/stack:output:0%gru_12/strided_slice/stack_1:output:0%gru_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_12/strided_slicej
gru_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_12/zeros/mul/y?
gru_12/zeros/mulMulgru_12/strided_slice:output:0gru_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_12/zeros/mulm
gru_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_12/zeros/Less/y?
gru_12/zeros/LessLessgru_12/zeros/mul:z:0gru_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_12/zeros/Lessp
gru_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_12/zeros/packed/1?
gru_12/zeros/packedPackgru_12/strided_slice:output:0gru_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_12/zeros/packedm
gru_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_12/zeros/Const?
gru_12/zerosFillgru_12/zeros/packed:output:0gru_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_12/zeros?
gru_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_12/transpose/perm?
gru_12/transpose	Transposerepeat_vector_12/Tile:output:0gru_12/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_12/transposed
gru_12/Shape_1Shapegru_12/transpose:y:0*
T0*
_output_shapes
:2
gru_12/Shape_1?
gru_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_12/strided_slice_1/stack?
gru_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_1/stack_1?
gru_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_1/stack_2?
gru_12/strided_slice_1StridedSlicegru_12/Shape_1:output:0%gru_12/strided_slice_1/stack:output:0'gru_12/strided_slice_1/stack_1:output:0'gru_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_12/strided_slice_1?
"gru_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_12/TensorArrayV2/element_shape?
gru_12/TensorArrayV2TensorListReserve+gru_12/TensorArrayV2/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_12/TensorArrayV2?
<gru_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<gru_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_12/transpose:y:0Egru_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_12/TensorArrayUnstack/TensorListFromTensor?
gru_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_12/strided_slice_2/stack?
gru_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_2/stack_1?
gru_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_2/stack_2?
gru_12/strided_slice_2StridedSlicegru_12/transpose:y:0%gru_12/strided_slice_2/stack:output:0'gru_12/strided_slice_2/stack_1:output:0'gru_12/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_12/strided_slice_2?
!gru_12/gru_cell_12/ReadVariableOpReadVariableOp*gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02#
!gru_12/gru_cell_12/ReadVariableOp?
&gru_12/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru_12/gru_cell_12/strided_slice/stack?
(gru_12/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(gru_12/gru_cell_12/strided_slice/stack_1?
(gru_12/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_12/gru_cell_12/strided_slice/stack_2?
 gru_12/gru_cell_12/strided_sliceStridedSlice)gru_12/gru_cell_12/ReadVariableOp:value:0/gru_12/gru_cell_12/strided_slice/stack:output:01gru_12/gru_cell_12/strided_slice/stack_1:output:01gru_12/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2"
 gru_12/gru_cell_12/strided_slice?
gru_12/gru_cell_12/MatMulMatMulgru_12/strided_slice_2:output:0)gru_12/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul?
#gru_12/gru_cell_12/ReadVariableOp_1ReadVariableOp*gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_1?
(gru_12/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru_12/gru_cell_12/strided_slice_1/stack?
*gru_12/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru_12/gru_cell_12/strided_slice_1/stack_1?
*gru_12/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_1/stack_2?
"gru_12/gru_cell_12/strided_slice_1StridedSlice+gru_12/gru_cell_12/ReadVariableOp_1:value:01gru_12/gru_cell_12/strided_slice_1/stack:output:03gru_12/gru_cell_12/strided_slice_1/stack_1:output:03gru_12/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_1?
gru_12/gru_cell_12/MatMul_1MatMulgru_12/strided_slice_2:output:0+gru_12/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_1?
#gru_12/gru_cell_12/ReadVariableOp_2ReadVariableOp*gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_2?
(gru_12/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru_12/gru_cell_12/strided_slice_2/stack?
*gru_12/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_12/gru_cell_12/strided_slice_2/stack_1?
*gru_12/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_2/stack_2?
"gru_12/gru_cell_12/strided_slice_2StridedSlice+gru_12/gru_cell_12/ReadVariableOp_2:value:01gru_12/gru_cell_12/strided_slice_2/stack:output:03gru_12/gru_cell_12/strided_slice_2/stack_1:output:03gru_12/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_2?
gru_12/gru_cell_12/MatMul_2MatMulgru_12/strided_slice_2:output:0+gru_12/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_2?
#gru_12/gru_cell_12/ReadVariableOp_3ReadVariableOp,gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_3?
(gru_12/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru_12/gru_cell_12/strided_slice_3/stack?
*gru_12/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_3/stack_1?
*gru_12/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_3/stack_2?
"gru_12/gru_cell_12/strided_slice_3StridedSlice+gru_12/gru_cell_12/ReadVariableOp_3:value:01gru_12/gru_cell_12/strided_slice_3/stack:output:03gru_12/gru_cell_12/strided_slice_3/stack_1:output:03gru_12/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"gru_12/gru_cell_12/strided_slice_3?
gru_12/gru_cell_12/BiasAddBiasAdd#gru_12/gru_cell_12/MatMul:product:0+gru_12/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/BiasAdd?
#gru_12/gru_cell_12/ReadVariableOp_4ReadVariableOp,gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_4?
(gru_12/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(gru_12/gru_cell_12/strided_slice_4/stack?
*gru_12/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02,
*gru_12/gru_cell_12/strided_slice_4/stack_1?
*gru_12/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_4/stack_2?
"gru_12/gru_cell_12/strided_slice_4StridedSlice+gru_12/gru_cell_12/ReadVariableOp_4:value:01gru_12/gru_cell_12/strided_slice_4/stack:output:03gru_12/gru_cell_12/strided_slice_4/stack_1:output:03gru_12/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2$
"gru_12/gru_cell_12/strided_slice_4?
gru_12/gru_cell_12/BiasAdd_1BiasAdd%gru_12/gru_cell_12/MatMul_1:product:0+gru_12/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/BiasAdd_1?
#gru_12/gru_cell_12/ReadVariableOp_5ReadVariableOp,gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_5?
(gru_12/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02*
(gru_12/gru_cell_12/strided_slice_5/stack?
*gru_12/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru_12/gru_cell_12/strided_slice_5/stack_1?
*gru_12/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru_12/gru_cell_12/strided_slice_5/stack_2?
"gru_12/gru_cell_12/strided_slice_5StridedSlice+gru_12/gru_cell_12/ReadVariableOp_5:value:01gru_12/gru_cell_12/strided_slice_5/stack:output:03gru_12/gru_cell_12/strided_slice_5/stack_1:output:03gru_12/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2$
"gru_12/gru_cell_12/strided_slice_5?
gru_12/gru_cell_12/BiasAdd_2BiasAdd%gru_12/gru_cell_12/MatMul_2:product:0+gru_12/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/BiasAdd_2?
#gru_12/gru_cell_12/ReadVariableOp_6ReadVariableOp,gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_6?
(gru_12/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru_12/gru_cell_12/strided_slice_6/stack?
*gru_12/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*gru_12/gru_cell_12/strided_slice_6/stack_1?
*gru_12/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_6/stack_2?
"gru_12/gru_cell_12/strided_slice_6StridedSlice+gru_12/gru_cell_12/ReadVariableOp_6:value:01gru_12/gru_cell_12/strided_slice_6/stack:output:03gru_12/gru_cell_12/strided_slice_6/stack_1:output:03gru_12/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_6?
gru_12/gru_cell_12/MatMul_3MatMulgru_12/zeros:output:0+gru_12/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_3?
#gru_12/gru_cell_12/ReadVariableOp_7ReadVariableOp,gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_7?
(gru_12/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(gru_12/gru_cell_12/strided_slice_7/stack?
*gru_12/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2,
*gru_12/gru_cell_12/strided_slice_7/stack_1?
*gru_12/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_7/stack_2?
"gru_12/gru_cell_12/strided_slice_7StridedSlice+gru_12/gru_cell_12/ReadVariableOp_7:value:01gru_12/gru_cell_12/strided_slice_7/stack:output:03gru_12/gru_cell_12/strided_slice_7/stack_1:output:03gru_12/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_7?
gru_12/gru_cell_12/MatMul_4MatMulgru_12/zeros:output:0+gru_12/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_4?
gru_12/gru_cell_12/addAddV2#gru_12/gru_cell_12/BiasAdd:output:0%gru_12/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/addy
gru_12/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_12/gru_cell_12/Const}
gru_12/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_12/gru_cell_12/Const_1?
gru_12/gru_cell_12/MulMulgru_12/gru_cell_12/add:z:0!gru_12/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Mul?
gru_12/gru_cell_12/Add_1AddV2gru_12/gru_cell_12/Mul:z:0#gru_12/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Add_1?
*gru_12/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*gru_12/gru_cell_12/clip_by_value/Minimum/y?
(gru_12/gru_cell_12/clip_by_value/MinimumMinimumgru_12/gru_cell_12/Add_1:z:03gru_12/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(gru_12/gru_cell_12/clip_by_value/Minimum?
"gru_12/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"gru_12/gru_cell_12/clip_by_value/y?
 gru_12/gru_cell_12/clip_by_valueMaximum,gru_12/gru_cell_12/clip_by_value/Minimum:z:0+gru_12/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 gru_12/gru_cell_12/clip_by_value?
gru_12/gru_cell_12/add_2AddV2%gru_12/gru_cell_12/BiasAdd_1:output:0%gru_12/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/add_2}
gru_12/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_12/gru_cell_12/Const_2}
gru_12/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_12/gru_cell_12/Const_3?
gru_12/gru_cell_12/Mul_1Mulgru_12/gru_cell_12/add_2:z:0#gru_12/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Mul_1?
gru_12/gru_cell_12/Add_3AddV2gru_12/gru_cell_12/Mul_1:z:0#gru_12/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Add_3?
,gru_12/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,gru_12/gru_cell_12/clip_by_value_1/Minimum/y?
*gru_12/gru_cell_12/clip_by_value_1/MinimumMinimumgru_12/gru_cell_12/Add_3:z:05gru_12/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2,
*gru_12/gru_cell_12/clip_by_value_1/Minimum?
$gru_12/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$gru_12/gru_cell_12/clip_by_value_1/y?
"gru_12/gru_cell_12/clip_by_value_1Maximum.gru_12/gru_cell_12/clip_by_value_1/Minimum:z:0-gru_12/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2$
"gru_12/gru_cell_12/clip_by_value_1?
gru_12/gru_cell_12/mul_2Mul&gru_12/gru_cell_12/clip_by_value_1:z:0gru_12/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/mul_2?
#gru_12/gru_cell_12/ReadVariableOp_8ReadVariableOp,gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02%
#gru_12/gru_cell_12/ReadVariableOp_8?
(gru_12/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2*
(gru_12/gru_cell_12/strided_slice_8/stack?
*gru_12/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_12/gru_cell_12/strided_slice_8/stack_1?
*gru_12/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru_12/gru_cell_12/strided_slice_8/stack_2?
"gru_12/gru_cell_12/strided_slice_8StridedSlice+gru_12/gru_cell_12/ReadVariableOp_8:value:01gru_12/gru_cell_12/strided_slice_8/stack:output:03gru_12/gru_cell_12/strided_slice_8/stack_1:output:03gru_12/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"gru_12/gru_cell_12/strided_slice_8?
gru_12/gru_cell_12/MatMul_5MatMulgru_12/gru_cell_12/mul_2:z:0+gru_12/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/MatMul_5?
gru_12/gru_cell_12/add_4AddV2%gru_12/gru_cell_12/BiasAdd_2:output:0%gru_12/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/add_4?
gru_12/gru_cell_12/ReluRelugru_12/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/Relu?
gru_12/gru_cell_12/mul_3Mul$gru_12/gru_cell_12/clip_by_value:z:0gru_12/zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/mul_3y
gru_12/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_12/gru_cell_12/sub/x?
gru_12/gru_cell_12/subSub!gru_12/gru_cell_12/sub/x:output:0$gru_12/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/sub?
gru_12/gru_cell_12/mul_4Mulgru_12/gru_cell_12/sub:z:0%gru_12/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/mul_4?
gru_12/gru_cell_12/add_5AddV2gru_12/gru_cell_12/mul_3:z:0gru_12/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_12/gru_cell_12/add_5?
$gru_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_12/TensorArrayV2_1/element_shape?
gru_12/TensorArrayV2_1TensorListReserve-gru_12/TensorArrayV2_1/element_shape:output:0gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_12/TensorArrayV2_1\
gru_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_12/time?
gru_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_12/while/maximum_iterationsx
gru_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_12/while/loop_counter?
gru_12/whileWhile"gru_12/while/loop_counter:output:0(gru_12/while/maximum_iterations:output:0gru_12/time:output:0gru_12/TensorArrayV2_1:handle:0gru_12/zeros:output:0gru_12/strided_slice_1:output:0>gru_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_12_gru_cell_12_readvariableop_resource,gru_12_gru_cell_12_readvariableop_3_resource,gru_12_gru_cell_12_readvariableop_6_resource*
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
_stateful_parallelism( *$
bodyR
gru_12_while_body_291311*$
condR
gru_12_while_cond_291310*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
gru_12/while?
7gru_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_12/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_12/TensorArrayV2Stack/TensorListStackTensorListStackgru_12/while:output:3@gru_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02+
)gru_12/TensorArrayV2Stack/TensorListStack?
gru_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_12/strided_slice_3/stack?
gru_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_12/strided_slice_3/stack_1?
gru_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_12/strided_slice_3/stack_2?
gru_12/strided_slice_3StridedSlice2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0%gru_12/strided_slice_3/stack:output:0'gru_12/strided_slice_3/stack_1:output:0'gru_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_12/strided_slice_3?
gru_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_12/transpose_1/perm?
gru_12/transpose_1	Transpose2gru_12/TensorArrayV2Stack/TensorListStack:tensor:0 gru_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_12/transpose_1y
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_12/dropout/Const?
dropout_12/dropout/MulMulgru_12/transpose_1:y:0!dropout_12/dropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout_12/dropout/Mulz
dropout_12/dropout/ShapeShapegru_12/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shape?
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform?
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_12/dropout/GreaterEqual/y?
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2!
dropout_12/dropout/GreaterEqual?
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout_12/dropout/Cast?
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout_12/dropout/Mul_1?
!time_distributed_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!time_distributed_24/Reshape/shape?
time_distributed_24/ReshapeReshapedropout_12/dropout/Mul_1:z:0*time_distributed_24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape?
2time_distributed_24/dense_24/MatMul/ReadVariableOpReadVariableOp;time_distributed_24_dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype024
2time_distributed_24/dense_24/MatMul/ReadVariableOp?
#time_distributed_24/dense_24/MatMulMatMul$time_distributed_24/Reshape:output:0:time_distributed_24/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#time_distributed_24/dense_24/MatMul?
3time_distributed_24/dense_24/BiasAdd/ReadVariableOpReadVariableOp<time_distributed_24_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3time_distributed_24/dense_24/BiasAdd/ReadVariableOp?
$time_distributed_24/dense_24/BiasAddBiasAdd-time_distributed_24/dense_24/MatMul:product:0;time_distributed_24/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$time_distributed_24/dense_24/BiasAdd?
#time_distributed_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2%
#time_distributed_24/Reshape_1/shape?
time_distributed_24/Reshape_1Reshape-time_distributed_24/dense_24/BiasAdd:output:0,time_distributed_24/Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_24/Reshape_1?
#time_distributed_24/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2%
#time_distributed_24/Reshape_2/shape?
time_distributed_24/Reshape_2Reshapedropout_12/dropout/Mul_1:z:0,time_distributed_24/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
time_distributed_24/Reshape_2?
!time_distributed_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!time_distributed_25/Reshape/shape?
time_distributed_25/ReshapeReshape&time_distributed_24/Reshape_1:output:0*time_distributed_25/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape?
2time_distributed_25/dense_25/MatMul/ReadVariableOpReadVariableOp;time_distributed_25_dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype024
2time_distributed_25/dense_25/MatMul/ReadVariableOp?
#time_distributed_25/dense_25/MatMulMatMul$time_distributed_25/Reshape:output:0:time_distributed_25/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#time_distributed_25/dense_25/MatMul?
3time_distributed_25/dense_25/BiasAdd/ReadVariableOpReadVariableOp<time_distributed_25_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3time_distributed_25/dense_25/BiasAdd/ReadVariableOp?
$time_distributed_25/dense_25/BiasAddBiasAdd-time_distributed_25/dense_25/MatMul:product:0;time_distributed_25/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$time_distributed_25/dense_25/BiasAdd?
#time_distributed_25/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2%
#time_distributed_25/Reshape_1/shape?
time_distributed_25/Reshape_1Reshape-time_distributed_25/dense_25/BiasAdd:output:0,time_distributed_25/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
time_distributed_25/Reshape_1?
#time_distributed_25/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2%
#time_distributed_25/Reshape_2/shape?
time_distributed_25/Reshape_2Reshape&time_distributed_24/Reshape_1:output:0,time_distributed_25/Reshape_2/shape:output:0*
T0*'
_output_shapes
:????????? 2
time_distributed_25/Reshape_2?
IdentityIdentity&time_distributed_25/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv1d_24/BiasAdd/ReadVariableOp-^conv1d_24/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_25/BiasAdd/ReadVariableOp-^conv1d_25/conv1d/ExpandDims_1/ReadVariableOp"^gru_12/gru_cell_12/ReadVariableOp$^gru_12/gru_cell_12/ReadVariableOp_1$^gru_12/gru_cell_12/ReadVariableOp_2$^gru_12/gru_cell_12/ReadVariableOp_3$^gru_12/gru_cell_12/ReadVariableOp_4$^gru_12/gru_cell_12/ReadVariableOp_5$^gru_12/gru_cell_12/ReadVariableOp_6$^gru_12/gru_cell_12/ReadVariableOp_7$^gru_12/gru_cell_12/ReadVariableOp_8^gru_12/while4^time_distributed_24/dense_24/BiasAdd/ReadVariableOp3^time_distributed_24/dense_24/MatMul/ReadVariableOp4^time_distributed_25/dense_25/BiasAdd/ReadVariableOp3^time_distributed_25/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2D
 conv1d_24/BiasAdd/ReadVariableOp conv1d_24/BiasAdd/ReadVariableOp2\
,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp,conv1d_24/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_25/BiasAdd/ReadVariableOp conv1d_25/BiasAdd/ReadVariableOp2\
,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp,conv1d_25/conv1d/ExpandDims_1/ReadVariableOp2F
!gru_12/gru_cell_12/ReadVariableOp!gru_12/gru_cell_12/ReadVariableOp2J
#gru_12/gru_cell_12/ReadVariableOp_1#gru_12/gru_cell_12/ReadVariableOp_12J
#gru_12/gru_cell_12/ReadVariableOp_2#gru_12/gru_cell_12/ReadVariableOp_22J
#gru_12/gru_cell_12/ReadVariableOp_3#gru_12/gru_cell_12/ReadVariableOp_32J
#gru_12/gru_cell_12/ReadVariableOp_4#gru_12/gru_cell_12/ReadVariableOp_42J
#gru_12/gru_cell_12/ReadVariableOp_5#gru_12/gru_cell_12/ReadVariableOp_52J
#gru_12/gru_cell_12/ReadVariableOp_6#gru_12/gru_cell_12/ReadVariableOp_62J
#gru_12/gru_cell_12/ReadVariableOp_7#gru_12/gru_cell_12/ReadVariableOp_72J
#gru_12/gru_cell_12/ReadVariableOp_8#gru_12/gru_cell_12/ReadVariableOp_82
gru_12/whilegru_12/while2j
3time_distributed_24/dense_24/BiasAdd/ReadVariableOp3time_distributed_24/dense_24/BiasAdd/ReadVariableOp2h
2time_distributed_24/dense_24/MatMul/ReadVariableOp2time_distributed_24/dense_24/MatMul/ReadVariableOp2j
3time_distributed_25/dense_25/BiasAdd/ReadVariableOp3time_distributed_25/dense_25/BiasAdd/ReadVariableOp2h
2time_distributed_25/dense_25/MatMul/ReadVariableOp2time_distributed_25/dense_25/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_gru_12_layer_call_and_return_conditional_losses_292150
inputs_06
#gru_cell_12_readvariableop_resource:	?H3
%gru_cell_12_readvariableop_3_resource:H7
%gru_cell_12_readvariableop_6_resource:H
identity??gru_cell_12/ReadVariableOp?gru_cell_12/ReadVariableOp_1?gru_cell_12/ReadVariableOp_2?gru_cell_12/ReadVariableOp_3?gru_cell_12/ReadVariableOp_4?gru_cell_12/ReadVariableOp_5?gru_cell_12/ReadVariableOp_6?gru_cell_12/ReadVariableOp_7?gru_cell_12/ReadVariableOp_8?whileF
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
gru_cell_12/ReadVariableOpReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp?
gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
gru_cell_12/strided_slice/stack?
!gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice/stack_1?
!gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell_12/strided_slice/stack_2?
gru_cell_12/strided_sliceStridedSlice"gru_cell_12/ReadVariableOp:value:0(gru_cell_12/strided_slice/stack:output:0*gru_cell_12/strided_slice/stack_1:output:0*gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice?
gru_cell_12/MatMulMatMulstrided_slice_2:output:0"gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul?
gru_cell_12/ReadVariableOp_1ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_1?
!gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_1/stack?
#gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_1/stack_1?
#gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_1/stack_2?
gru_cell_12/strided_slice_1StridedSlice$gru_cell_12/ReadVariableOp_1:value:0*gru_cell_12/strided_slice_1/stack:output:0,gru_cell_12/strided_slice_1/stack_1:output:0,gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_1?
gru_cell_12/MatMul_1MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_1?
gru_cell_12/ReadVariableOp_2ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_2?
!gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_2/stack?
#gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_2/stack_1?
#gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_2/stack_2?
gru_cell_12/strided_slice_2StridedSlice$gru_cell_12/ReadVariableOp_2:value:0*gru_cell_12/strided_slice_2/stack:output:0,gru_cell_12/strided_slice_2/stack_1:output:0,gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_2?
gru_cell_12/MatMul_2MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_2?
gru_cell_12/ReadVariableOp_3ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_3?
!gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell_12/strided_slice_3/stack?
#gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_1?
#gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_2?
gru_cell_12/strided_slice_3StridedSlice$gru_cell_12/ReadVariableOp_3:value:0*gru_cell_12/strided_slice_3/stack:output:0,gru_cell_12/strided_slice_3/stack_1:output:0,gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell_12/strided_slice_3?
gru_cell_12/BiasAddBiasAddgru_cell_12/MatMul:product:0$gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd?
gru_cell_12/ReadVariableOp_4ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_4?
!gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell_12/strided_slice_4/stack?
#gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02%
#gru_cell_12/strided_slice_4/stack_1?
#gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_4/stack_2?
gru_cell_12/strided_slice_4StridedSlice$gru_cell_12/ReadVariableOp_4:value:0*gru_cell_12/strided_slice_4/stack:output:0,gru_cell_12/strided_slice_4/stack_1:output:0,gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell_12/strided_slice_4?
gru_cell_12/BiasAdd_1BiasAddgru_cell_12/MatMul_1:product:0$gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_1?
gru_cell_12/ReadVariableOp_5ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_5?
!gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02#
!gru_cell_12/strided_slice_5/stack?
#gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_12/strided_slice_5/stack_1?
#gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_5/stack_2?
gru_cell_12/strided_slice_5StridedSlice$gru_cell_12/ReadVariableOp_5:value:0*gru_cell_12/strided_slice_5/stack:output:0,gru_cell_12/strided_slice_5/stack_1:output:0,gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell_12/strided_slice_5?
gru_cell_12/BiasAdd_2BiasAddgru_cell_12/MatMul_2:product:0$gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_2?
gru_cell_12/ReadVariableOp_6ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_6?
!gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell_12/strided_slice_6/stack?
#gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#gru_cell_12/strided_slice_6/stack_1?
#gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_6/stack_2?
gru_cell_12/strided_slice_6StridedSlice$gru_cell_12/ReadVariableOp_6:value:0*gru_cell_12/strided_slice_6/stack:output:0,gru_cell_12/strided_slice_6/stack_1:output:0,gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_6?
gru_cell_12/MatMul_3MatMulzeros:output:0$gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_3?
gru_cell_12/ReadVariableOp_7ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_7?
!gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_7/stack?
#gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_7/stack_1?
#gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_7/stack_2?
gru_cell_12/strided_slice_7StridedSlice$gru_cell_12/ReadVariableOp_7:value:0*gru_cell_12/strided_slice_7/stack:output:0,gru_cell_12/strided_slice_7/stack_1:output:0,gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_7?
gru_cell_12/MatMul_4MatMulzeros:output:0$gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_4?
gru_cell_12/addAddV2gru_cell_12/BiasAdd:output:0gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/addk
gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Consto
gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_1?
gru_cell_12/MulMulgru_cell_12/add:z:0gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul?
gru_cell_12/Add_1AddV2gru_cell_12/Mul:z:0gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_1?
#gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#gru_cell_12/clip_by_value/Minimum/y?
!gru_cell_12/clip_by_value/MinimumMinimumgru_cell_12/Add_1:z:0,gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2#
!gru_cell_12/clip_by_value/Minimum
gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value/y?
gru_cell_12/clip_by_valueMaximum%gru_cell_12/clip_by_value/Minimum:z:0$gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value?
gru_cell_12/add_2AddV2gru_cell_12/BiasAdd_1:output:0gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_2o
gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Const_2o
gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_3?
gru_cell_12/Mul_1Mulgru_cell_12/add_2:z:0gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul_1?
gru_cell_12/Add_3AddV2gru_cell_12/Mul_1:z:0gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_3?
%gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gru_cell_12/clip_by_value_1/Minimum/y?
#gru_cell_12/clip_by_value_1/MinimumMinimumgru_cell_12/Add_3:z:0.gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2%
#gru_cell_12/clip_by_value_1/Minimum?
gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value_1/y?
gru_cell_12/clip_by_value_1Maximum'gru_cell_12/clip_by_value_1/Minimum:z:0&gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value_1?
gru_cell_12/mul_2Mulgru_cell_12/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_2?
gru_cell_12/ReadVariableOp_8ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_8?
!gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_8/stack?
#gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_8/stack_1?
#gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_8/stack_2?
gru_cell_12/strided_slice_8StridedSlice$gru_cell_12/ReadVariableOp_8:value:0*gru_cell_12/strided_slice_8/stack:output:0,gru_cell_12/strided_slice_8/stack_1:output:0,gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_8?
gru_cell_12/MatMul_5MatMulgru_cell_12/mul_2:z:0$gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_5?
gru_cell_12/add_4AddV2gru_cell_12/BiasAdd_2:output:0gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_4u
gru_cell_12/ReluRelugru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Relu?
gru_cell_12/mul_3Mulgru_cell_12/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_3k
gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_12/sub/x?
gru_cell_12/subSubgru_cell_12/sub/x:output:0gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/sub?
gru_cell_12/mul_4Mulgru_cell_12/sub:z:0gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_4?
gru_cell_12/add_5AddV2gru_cell_12/mul_3:z:0gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_5?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_12_readvariableop_resource%gru_cell_12_readvariableop_3_resource%gru_cell_12_readvariableop_6_resource*
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
bodyR
while_body_292012*
condR
while_cond_292011*8
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
NoOpNoOp^gru_cell_12/ReadVariableOp^gru_cell_12/ReadVariableOp_1^gru_cell_12/ReadVariableOp_2^gru_cell_12/ReadVariableOp_3^gru_cell_12/ReadVariableOp_4^gru_cell_12/ReadVariableOp_5^gru_cell_12/ReadVariableOp_6^gru_cell_12/ReadVariableOp_7^gru_cell_12/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 28
gru_cell_12/ReadVariableOpgru_cell_12/ReadVariableOp2<
gru_cell_12/ReadVariableOp_1gru_cell_12/ReadVariableOp_12<
gru_cell_12/ReadVariableOp_2gru_cell_12/ReadVariableOp_22<
gru_cell_12/ReadVariableOp_3gru_cell_12/ReadVariableOp_32<
gru_cell_12/ReadVariableOp_4gru_cell_12/ReadVariableOp_42<
gru_cell_12/ReadVariableOp_5gru_cell_12/ReadVariableOp_52<
gru_cell_12/ReadVariableOp_6gru_cell_12/ReadVariableOp_62<
gru_cell_12/ReadVariableOp_7gru_cell_12/ReadVariableOp_72<
gru_cell_12/ReadVariableOp_8gru_cell_12/ReadVariableOp_82
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_289685

inputs!
dense_25_289675: 
dense_25_289677:
identity?? dense_25/StatefulPartitionedCallD
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
 dense_25/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_25_289675dense_25_289677*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_2896262"
 dense_25/StatefulPartitionedCallq
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
	Reshape_1Reshape)dense_25/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityq
NoOpNoOp!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
h
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_289816

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
h
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_291557

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
4__inference_time_distributed_25_layer_call_fn_292813

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2896852
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
?
h
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_291549

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
?]
?
__inference__traced_save_293297
file_prefix/
+savev2_conv1d_24_kernel_read_readvariableop-
)savev2_conv1d_24_bias_read_readvariableop/
+savev2_conv1d_25_kernel_read_readvariableop-
)savev2_conv1d_25_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop8
4savev2_gru_12_gru_cell_12_kernel_read_readvariableopB
>savev2_gru_12_gru_cell_12_recurrent_kernel_read_readvariableop6
2savev2_gru_12_gru_cell_12_bias_read_readvariableop9
5savev2_time_distributed_24_kernel_read_readvariableop7
3savev2_time_distributed_24_bias_read_readvariableop9
5savev2_time_distributed_25_kernel_read_readvariableop7
3savev2_time_distributed_25_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_nadam_conv1d_24_kernel_m_read_readvariableop5
1savev2_nadam_conv1d_24_bias_m_read_readvariableop7
3savev2_nadam_conv1d_25_kernel_m_read_readvariableop5
1savev2_nadam_conv1d_25_bias_m_read_readvariableop@
<savev2_nadam_gru_12_gru_cell_12_kernel_m_read_readvariableopJ
Fsavev2_nadam_gru_12_gru_cell_12_recurrent_kernel_m_read_readvariableop>
:savev2_nadam_gru_12_gru_cell_12_bias_m_read_readvariableopA
=savev2_nadam_time_distributed_24_kernel_m_read_readvariableop?
;savev2_nadam_time_distributed_24_bias_m_read_readvariableopA
=savev2_nadam_time_distributed_25_kernel_m_read_readvariableop?
;savev2_nadam_time_distributed_25_bias_m_read_readvariableop7
3savev2_nadam_conv1d_24_kernel_v_read_readvariableop5
1savev2_nadam_conv1d_24_bias_v_read_readvariableop7
3savev2_nadam_conv1d_25_kernel_v_read_readvariableop5
1savev2_nadam_conv1d_25_bias_v_read_readvariableop@
<savev2_nadam_gru_12_gru_cell_12_kernel_v_read_readvariableopJ
Fsavev2_nadam_gru_12_gru_cell_12_recurrent_kernel_v_read_readvariableop>
:savev2_nadam_gru_12_gru_cell_12_bias_v_read_readvariableopA
=savev2_nadam_time_distributed_24_kernel_v_read_readvariableop?
;savev2_nadam_time_distributed_24_bias_v_read_readvariableopA
=savev2_nadam_time_distributed_25_kernel_v_read_readvariableop?
;savev2_nadam_time_distributed_25_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_24_kernel_read_readvariableop)savev2_conv1d_24_bias_read_readvariableop+savev2_conv1d_25_kernel_read_readvariableop)savev2_conv1d_25_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop4savev2_gru_12_gru_cell_12_kernel_read_readvariableop>savev2_gru_12_gru_cell_12_recurrent_kernel_read_readvariableop2savev2_gru_12_gru_cell_12_bias_read_readvariableop5savev2_time_distributed_24_kernel_read_readvariableop3savev2_time_distributed_24_bias_read_readvariableop5savev2_time_distributed_25_kernel_read_readvariableop3savev2_time_distributed_25_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_nadam_conv1d_24_kernel_m_read_readvariableop1savev2_nadam_conv1d_24_bias_m_read_readvariableop3savev2_nadam_conv1d_25_kernel_m_read_readvariableop1savev2_nadam_conv1d_25_bias_m_read_readvariableop<savev2_nadam_gru_12_gru_cell_12_kernel_m_read_readvariableopFsavev2_nadam_gru_12_gru_cell_12_recurrent_kernel_m_read_readvariableop:savev2_nadam_gru_12_gru_cell_12_bias_m_read_readvariableop=savev2_nadam_time_distributed_24_kernel_m_read_readvariableop;savev2_nadam_time_distributed_24_bias_m_read_readvariableop=savev2_nadam_time_distributed_25_kernel_m_read_readvariableop;savev2_nadam_time_distributed_25_bias_m_read_readvariableop3savev2_nadam_conv1d_24_kernel_v_read_readvariableop1savev2_nadam_conv1d_24_bias_v_read_readvariableop3savev2_nadam_conv1d_25_kernel_v_read_readvariableop1savev2_nadam_conv1d_25_bias_v_read_readvariableop<savev2_nadam_gru_12_gru_cell_12_kernel_v_read_readvariableopFsavev2_nadam_gru_12_gru_cell_12_recurrent_kernel_v_read_readvariableop:savev2_nadam_gru_12_gru_cell_12_bias_v_read_readvariableop=savev2_nadam_time_distributed_24_kernel_v_read_readvariableop;savev2_nadam_time_distributed_24_bias_v_read_readvariableop=savev2_nadam_time_distributed_25_kernel_v_read_readvariableop;savev2_nadam_time_distributed_25_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?

?
$__inference_signature_wrapper_290790
conv1d_24_input
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
StatefulPartitionedCallStatefulPartitionedCallconv1d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
GPU2*0J 8? **
f%R#
!__inference__wrapped_model_2887292
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
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameconv1d_24_input
?
?
.__inference_sequential_12_layer_call_fn_290844

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
GPU2*0J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_2906252
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
?
?
'__inference_gru_12_layer_call_fn_291605
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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2889852
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
??
?	
while_body_290374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_12_readvariableop_resource_0:	?H;
-while_gru_cell_12_readvariableop_3_resource_0:H?
-while_gru_cell_12_readvariableop_6_resource_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_12_readvariableop_resource:	?H9
+while_gru_cell_12_readvariableop_3_resource:H=
+while_gru_cell_12_readvariableop_6_resource:H?? while/gru_cell_12/ReadVariableOp?"while/gru_cell_12/ReadVariableOp_1?"while/gru_cell_12/ReadVariableOp_2?"while/gru_cell_12/ReadVariableOp_3?"while/gru_cell_12/ReadVariableOp_4?"while/gru_cell_12/ReadVariableOp_5?"while/gru_cell_12/ReadVariableOp_6?"while/gru_cell_12/ReadVariableOp_7?"while/gru_cell_12/ReadVariableOp_8?
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
 while/gru_cell_12/ReadVariableOpReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02"
 while/gru_cell_12/ReadVariableOp?
%while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/gru_cell_12/strided_slice/stack?
'while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice/stack_1?
'while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell_12/strided_slice/stack_2?
while/gru_cell_12/strided_sliceStridedSlice(while/gru_cell_12/ReadVariableOp:value:0.while/gru_cell_12/strided_slice/stack:output:00while/gru_cell_12/strided_slice/stack_1:output:00while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2!
while/gru_cell_12/strided_slice?
while/gru_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0(while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul?
"while/gru_cell_12/ReadVariableOp_1ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_1?
'while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_1/stack?
)while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_1/stack_1?
)while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_1/stack_2?
!while/gru_cell_12/strided_slice_1StridedSlice*while/gru_cell_12/ReadVariableOp_1:value:00while/gru_cell_12/strided_slice_1/stack:output:02while/gru_cell_12/strided_slice_1/stack_1:output:02while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_1?
while/gru_cell_12/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_1?
"while/gru_cell_12/ReadVariableOp_2ReadVariableOp+while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02$
"while/gru_cell_12/ReadVariableOp_2?
'while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_2/stack?
)while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_2/stack_1?
)while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_2/stack_2?
!while/gru_cell_12/strided_slice_2StridedSlice*while/gru_cell_12/ReadVariableOp_2:value:00while/gru_cell_12/strided_slice_2/stack:output:02while/gru_cell_12/strided_slice_2/stack_1:output:02while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_2?
while/gru_cell_12/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_2?
"while/gru_cell_12/ReadVariableOp_3ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_3?
'while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell_12/strided_slice_3/stack?
)while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_1?
)while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_3/stack_2?
!while/gru_cell_12/strided_slice_3StridedSlice*while/gru_cell_12/ReadVariableOp_3:value:00while/gru_cell_12/strided_slice_3/stack:output:02while/gru_cell_12/strided_slice_3/stack_1:output:02while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2#
!while/gru_cell_12/strided_slice_3?
while/gru_cell_12/BiasAddBiasAdd"while/gru_cell_12/MatMul:product:0*while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd?
"while/gru_cell_12/ReadVariableOp_4ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_4?
'while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell_12/strided_slice_4/stack?
)while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02+
)while/gru_cell_12/strided_slice_4/stack_1?
)while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_4/stack_2?
!while/gru_cell_12/strided_slice_4StridedSlice*while/gru_cell_12/ReadVariableOp_4:value:00while/gru_cell_12/strided_slice_4/stack:output:02while/gru_cell_12/strided_slice_4/stack_1:output:02while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!while/gru_cell_12/strided_slice_4?
while/gru_cell_12/BiasAdd_1BiasAdd$while/gru_cell_12/MatMul_1:product:0*while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_1?
"while/gru_cell_12/ReadVariableOp_5ReadVariableOp-while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_5?
'while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02)
'while/gru_cell_12/strided_slice_5/stack?
)while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_12/strided_slice_5/stack_1?
)while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_12/strided_slice_5/stack_2?
!while/gru_cell_12/strided_slice_5StridedSlice*while/gru_cell_12/ReadVariableOp_5:value:00while/gru_cell_12/strided_slice_5/stack:output:02while/gru_cell_12/strided_slice_5/stack_1:output:02while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2#
!while/gru_cell_12/strided_slice_5?
while/gru_cell_12/BiasAdd_2BiasAdd$while/gru_cell_12/MatMul_2:product:0*while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/BiasAdd_2?
"while/gru_cell_12/ReadVariableOp_6ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_6?
'while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell_12/strided_slice_6/stack?
)while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/gru_cell_12/strided_slice_6/stack_1?
)while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_6/stack_2?
!while/gru_cell_12/strided_slice_6StridedSlice*while/gru_cell_12/ReadVariableOp_6:value:00while/gru_cell_12/strided_slice_6/stack:output:02while/gru_cell_12/strided_slice_6/stack_1:output:02while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_6?
while/gru_cell_12/MatMul_3MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_3?
"while/gru_cell_12/ReadVariableOp_7ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_7?
'while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/gru_cell_12/strided_slice_7/stack?
)while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2+
)while/gru_cell_12/strided_slice_7/stack_1?
)while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_7/stack_2?
!while/gru_cell_12/strided_slice_7StridedSlice*while/gru_cell_12/ReadVariableOp_7:value:00while/gru_cell_12/strided_slice_7/stack:output:02while/gru_cell_12/strided_slice_7/stack_1:output:02while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_7?
while/gru_cell_12/MatMul_4MatMulwhile_placeholder_2*while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_4?
while/gru_cell_12/addAddV2"while/gru_cell_12/BiasAdd:output:0$while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/addw
while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const{
while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_1?
while/gru_cell_12/MulMulwhile/gru_cell_12/add:z:0 while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul?
while/gru_cell_12/Add_1AddV2while/gru_cell_12/Mul:z:0"while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_1?
)while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/gru_cell_12/clip_by_value/Minimum/y?
'while/gru_cell_12/clip_by_value/MinimumMinimumwhile/gru_cell_12/Add_1:z:02while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2)
'while/gru_cell_12/clip_by_value/Minimum?
!while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/gru_cell_12/clip_by_value/y?
while/gru_cell_12/clip_by_valueMaximum+while/gru_cell_12/clip_by_value/Minimum:z:0*while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2!
while/gru_cell_12/clip_by_value?
while/gru_cell_12/add_2AddV2$while/gru_cell_12/BiasAdd_1:output:0$while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_2{
while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/gru_cell_12/Const_2{
while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/gru_cell_12/Const_3?
while/gru_cell_12/Mul_1Mulwhile/gru_cell_12/add_2:z:0"while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Mul_1?
while/gru_cell_12/Add_3AddV2while/gru_cell_12/Mul_1:z:0"while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Add_3?
+while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+while/gru_cell_12/clip_by_value_1/Minimum/y?
)while/gru_cell_12/clip_by_value_1/MinimumMinimumwhile/gru_cell_12/Add_3:z:04while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)while/gru_cell_12/clip_by_value_1/Minimum?
#while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#while/gru_cell_12/clip_by_value_1/y?
!while/gru_cell_12/clip_by_value_1Maximum-while/gru_cell_12/clip_by_value_1/Minimum:z:0,while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2#
!while/gru_cell_12/clip_by_value_1?
while/gru_cell_12/mul_2Mul%while/gru_cell_12/clip_by_value_1:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_2?
"while/gru_cell_12/ReadVariableOp_8ReadVariableOp-while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02$
"while/gru_cell_12/ReadVariableOp_8?
'while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2)
'while/gru_cell_12/strided_slice_8/stack?
)while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_12/strided_slice_8/stack_1?
)while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_12/strided_slice_8/stack_2?
!while/gru_cell_12/strided_slice_8StridedSlice*while/gru_cell_12/ReadVariableOp_8:value:00while/gru_cell_12/strided_slice_8/stack:output:02while/gru_cell_12/strided_slice_8/stack_1:output:02while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/gru_cell_12/strided_slice_8?
while/gru_cell_12/MatMul_5MatMulwhile/gru_cell_12/mul_2:z:0*while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/MatMul_5?
while/gru_cell_12/add_4AddV2$while/gru_cell_12/BiasAdd_2:output:0$while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_4?
while/gru_cell_12/ReluReluwhile/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/Relu?
while/gru_cell_12/mul_3Mul#while/gru_cell_12/clip_by_value:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_3w
while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_12/sub/x?
while/gru_cell_12/subSub while/gru_cell_12/sub/x:output:0#while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/sub?
while/gru_cell_12/mul_4Mulwhile/gru_cell_12/sub:z:0$while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/mul_4?
while/gru_cell_12/add_5AddV2while/gru_cell_12/mul_3:z:0while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
while/gru_cell_12/add_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_12/add_5:z:0*
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
while/Identity_4Identitywhile/gru_cell_12/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp!^while/gru_cell_12/ReadVariableOp#^while/gru_cell_12/ReadVariableOp_1#^while/gru_cell_12/ReadVariableOp_2#^while/gru_cell_12/ReadVariableOp_3#^while/gru_cell_12/ReadVariableOp_4#^while/gru_cell_12/ReadVariableOp_5#^while/gru_cell_12/ReadVariableOp_6#^while/gru_cell_12/ReadVariableOp_7#^while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"\
+while_gru_cell_12_readvariableop_3_resource-while_gru_cell_12_readvariableop_3_resource_0"\
+while_gru_cell_12_readvariableop_6_resource-while_gru_cell_12_readvariableop_6_resource_0"X
)while_gru_cell_12_readvariableop_resource+while_gru_cell_12_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2D
 while/gru_cell_12/ReadVariableOp while/gru_cell_12/ReadVariableOp2H
"while/gru_cell_12/ReadVariableOp_1"while/gru_cell_12/ReadVariableOp_12H
"while/gru_cell_12/ReadVariableOp_2"while/gru_cell_12/ReadVariableOp_22H
"while/gru_cell_12/ReadVariableOp_3"while/gru_cell_12/ReadVariableOp_32H
"while/gru_cell_12/ReadVariableOp_4"while/gru_cell_12/ReadVariableOp_42H
"while/gru_cell_12/ReadVariableOp_5"while/gru_cell_12/ReadVariableOp_52H
"while/gru_cell_12/ReadVariableOp_6"while/gru_cell_12/ReadVariableOp_62H
"while/gru_cell_12/ReadVariableOp_7"while/gru_cell_12/ReadVariableOp_72H
"while/gru_cell_12/ReadVariableOp_8"while/gru_cell_12/ReadVariableOp_8: 
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
while_cond_288921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_288921___redundant_placeholder04
0while_while_cond_288921___redundant_placeholder14
0while_while_cond_288921___redundant_placeholder24
0while_while_cond_288921___redundant_placeholder3
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
?
d
+__inference_dropout_12_layer_call_fn_292672

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
GPU2*0J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2902402
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
?"
?
while_body_289168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_12_289190_0:	?H(
while_gru_cell_12_289192_0:H,
while_gru_cell_12_289194_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_12_289190:	?H&
while_gru_cell_12_289192:H*
while_gru_cell_12_289194:H??)while/gru_cell_12/StatefulPartitionedCall?
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
)while/gru_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_12_289190_0while_gru_cell_12_289192_0while_gru_cell_12_289194_0*
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
GPU2*0J 8? *P
fKRI
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_2891012+
)while/gru_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_12/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"6
while_gru_cell_12_289190while_gru_cell_12_289190_0"6
while_gru_cell_12_289192while_gru_cell_12_289192_0"6
while_gru_cell_12_289194while_gru_cell_12_289194_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2V
)while/gru_cell_12/StatefulPartitionedCall)while/gru_cell_12/StatefulPartitionedCall: 
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
4__inference_time_distributed_25_layer_call_fn_292822

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
GPU2*0J 8? *X
fSRQ
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_2901222
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
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292767

inputs9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource: 
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOpD
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
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulReshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/BiasAddq
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
	Reshape_1Reshapedense_24/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
'__inference_gru_12_layer_call_fn_291627

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
GPU2*0J 8? *K
fFRD
B__inference_gru_12_layer_call_and_return_conditional_losses_2900732
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
?
M
1__inference_repeat_vector_12_layer_call_fn_291578

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
GPU2*0J 8? *U
fPRN
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_2898162
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
?
?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292873

inputs9
'dense_25_matmul_readvariableop_resource: 6
(dense_25_biasadd_readvariableop_resource:
identity??dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOpD
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
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMulReshape:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAddq
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
	Reshape_1Reshapedense_25/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????????????? : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
??
?
B__inference_gru_12_layer_call_and_return_conditional_losses_290512

inputs6
#gru_cell_12_readvariableop_resource:	?H3
%gru_cell_12_readvariableop_3_resource:H7
%gru_cell_12_readvariableop_6_resource:H
identity??gru_cell_12/ReadVariableOp?gru_cell_12/ReadVariableOp_1?gru_cell_12/ReadVariableOp_2?gru_cell_12/ReadVariableOp_3?gru_cell_12/ReadVariableOp_4?gru_cell_12/ReadVariableOp_5?gru_cell_12/ReadVariableOp_6?gru_cell_12/ReadVariableOp_7?gru_cell_12/ReadVariableOp_8?whileD
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
gru_cell_12/ReadVariableOpReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp?
gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
gru_cell_12/strided_slice/stack?
!gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice/stack_1?
!gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell_12/strided_slice/stack_2?
gru_cell_12/strided_sliceStridedSlice"gru_cell_12/ReadVariableOp:value:0(gru_cell_12/strided_slice/stack:output:0*gru_cell_12/strided_slice/stack_1:output:0*gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice?
gru_cell_12/MatMulMatMulstrided_slice_2:output:0"gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul?
gru_cell_12/ReadVariableOp_1ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_1?
!gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_1/stack?
#gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_1/stack_1?
#gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_1/stack_2?
gru_cell_12/strided_slice_1StridedSlice$gru_cell_12/ReadVariableOp_1:value:0*gru_cell_12/strided_slice_1/stack:output:0,gru_cell_12/strided_slice_1/stack_1:output:0,gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_1?
gru_cell_12/MatMul_1MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_1?
gru_cell_12/ReadVariableOp_2ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_2?
!gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_2/stack?
#gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_2/stack_1?
#gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_2/stack_2?
gru_cell_12/strided_slice_2StridedSlice$gru_cell_12/ReadVariableOp_2:value:0*gru_cell_12/strided_slice_2/stack:output:0,gru_cell_12/strided_slice_2/stack_1:output:0,gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_2?
gru_cell_12/MatMul_2MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_2?
gru_cell_12/ReadVariableOp_3ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_3?
!gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell_12/strided_slice_3/stack?
#gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_1?
#gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_2?
gru_cell_12/strided_slice_3StridedSlice$gru_cell_12/ReadVariableOp_3:value:0*gru_cell_12/strided_slice_3/stack:output:0,gru_cell_12/strided_slice_3/stack_1:output:0,gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell_12/strided_slice_3?
gru_cell_12/BiasAddBiasAddgru_cell_12/MatMul:product:0$gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd?
gru_cell_12/ReadVariableOp_4ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_4?
!gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell_12/strided_slice_4/stack?
#gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02%
#gru_cell_12/strided_slice_4/stack_1?
#gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_4/stack_2?
gru_cell_12/strided_slice_4StridedSlice$gru_cell_12/ReadVariableOp_4:value:0*gru_cell_12/strided_slice_4/stack:output:0,gru_cell_12/strided_slice_4/stack_1:output:0,gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell_12/strided_slice_4?
gru_cell_12/BiasAdd_1BiasAddgru_cell_12/MatMul_1:product:0$gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_1?
gru_cell_12/ReadVariableOp_5ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_5?
!gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02#
!gru_cell_12/strided_slice_5/stack?
#gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_12/strided_slice_5/stack_1?
#gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_5/stack_2?
gru_cell_12/strided_slice_5StridedSlice$gru_cell_12/ReadVariableOp_5:value:0*gru_cell_12/strided_slice_5/stack:output:0,gru_cell_12/strided_slice_5/stack_1:output:0,gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell_12/strided_slice_5?
gru_cell_12/BiasAdd_2BiasAddgru_cell_12/MatMul_2:product:0$gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_2?
gru_cell_12/ReadVariableOp_6ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_6?
!gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell_12/strided_slice_6/stack?
#gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#gru_cell_12/strided_slice_6/stack_1?
#gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_6/stack_2?
gru_cell_12/strided_slice_6StridedSlice$gru_cell_12/ReadVariableOp_6:value:0*gru_cell_12/strided_slice_6/stack:output:0,gru_cell_12/strided_slice_6/stack_1:output:0,gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_6?
gru_cell_12/MatMul_3MatMulzeros:output:0$gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_3?
gru_cell_12/ReadVariableOp_7ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_7?
!gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_7/stack?
#gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_7/stack_1?
#gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_7/stack_2?
gru_cell_12/strided_slice_7StridedSlice$gru_cell_12/ReadVariableOp_7:value:0*gru_cell_12/strided_slice_7/stack:output:0,gru_cell_12/strided_slice_7/stack_1:output:0,gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_7?
gru_cell_12/MatMul_4MatMulzeros:output:0$gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_4?
gru_cell_12/addAddV2gru_cell_12/BiasAdd:output:0gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/addk
gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Consto
gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_1?
gru_cell_12/MulMulgru_cell_12/add:z:0gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul?
gru_cell_12/Add_1AddV2gru_cell_12/Mul:z:0gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_1?
#gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#gru_cell_12/clip_by_value/Minimum/y?
!gru_cell_12/clip_by_value/MinimumMinimumgru_cell_12/Add_1:z:0,gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2#
!gru_cell_12/clip_by_value/Minimum
gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value/y?
gru_cell_12/clip_by_valueMaximum%gru_cell_12/clip_by_value/Minimum:z:0$gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value?
gru_cell_12/add_2AddV2gru_cell_12/BiasAdd_1:output:0gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_2o
gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Const_2o
gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_3?
gru_cell_12/Mul_1Mulgru_cell_12/add_2:z:0gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul_1?
gru_cell_12/Add_3AddV2gru_cell_12/Mul_1:z:0gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_3?
%gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gru_cell_12/clip_by_value_1/Minimum/y?
#gru_cell_12/clip_by_value_1/MinimumMinimumgru_cell_12/Add_3:z:0.gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2%
#gru_cell_12/clip_by_value_1/Minimum?
gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value_1/y?
gru_cell_12/clip_by_value_1Maximum'gru_cell_12/clip_by_value_1/Minimum:z:0&gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value_1?
gru_cell_12/mul_2Mulgru_cell_12/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_2?
gru_cell_12/ReadVariableOp_8ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_8?
!gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_8/stack?
#gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_8/stack_1?
#gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_8/stack_2?
gru_cell_12/strided_slice_8StridedSlice$gru_cell_12/ReadVariableOp_8:value:0*gru_cell_12/strided_slice_8/stack:output:0,gru_cell_12/strided_slice_8/stack_1:output:0,gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_8?
gru_cell_12/MatMul_5MatMulgru_cell_12/mul_2:z:0$gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_5?
gru_cell_12/add_4AddV2gru_cell_12/BiasAdd_2:output:0gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_4u
gru_cell_12/ReluRelugru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Relu?
gru_cell_12/mul_3Mulgru_cell_12/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_3k
gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_12/sub/x?
gru_cell_12/subSubgru_cell_12/sub/x:output:0gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/sub?
gru_cell_12/mul_4Mulgru_cell_12/sub:z:0gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_4?
gru_cell_12/add_5AddV2gru_cell_12/mul_3:z:0gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_5?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_12_readvariableop_resource%gru_cell_12_readvariableop_3_resource%gru_cell_12_readvariableop_6_resource*
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
bodyR
while_body_290374*
condR
while_cond_290373*8
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
NoOpNoOp^gru_cell_12/ReadVariableOp^gru_cell_12/ReadVariableOp_1^gru_cell_12/ReadVariableOp_2^gru_cell_12/ReadVariableOp_3^gru_cell_12/ReadVariableOp_4^gru_cell_12/ReadVariableOp_5^gru_cell_12/ReadVariableOp_6^gru_cell_12/ReadVariableOp_7^gru_cell_12/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
gru_cell_12/ReadVariableOpgru_cell_12/ReadVariableOp2<
gru_cell_12/ReadVariableOp_1gru_cell_12/ReadVariableOp_12<
gru_cell_12/ReadVariableOp_2gru_cell_12/ReadVariableOp_22<
gru_cell_12/ReadVariableOp_3gru_cell_12/ReadVariableOp_32<
gru_cell_12/ReadVariableOp_4gru_cell_12/ReadVariableOp_42<
gru_cell_12/ReadVariableOp_5gru_cell_12/ReadVariableOp_52<
gru_cell_12/ReadVariableOp_6gru_cell_12/ReadVariableOp_62<
gru_cell_12/ReadVariableOp_7gru_cell_12/ReadVariableOp_72<
gru_cell_12/ReadVariableOp_8gru_cell_12/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
gru_12_while_body_291311*
&gru_12_while_gru_12_while_loop_counter0
,gru_12_while_gru_12_while_maximum_iterations
gru_12_while_placeholder
gru_12_while_placeholder_1
gru_12_while_placeholder_2)
%gru_12_while_gru_12_strided_slice_1_0e
agru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0E
2gru_12_while_gru_cell_12_readvariableop_resource_0:	?HB
4gru_12_while_gru_cell_12_readvariableop_3_resource_0:HF
4gru_12_while_gru_cell_12_readvariableop_6_resource_0:H
gru_12_while_identity
gru_12_while_identity_1
gru_12_while_identity_2
gru_12_while_identity_3
gru_12_while_identity_4'
#gru_12_while_gru_12_strided_slice_1c
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensorC
0gru_12_while_gru_cell_12_readvariableop_resource:	?H@
2gru_12_while_gru_cell_12_readvariableop_3_resource:HD
2gru_12_while_gru_cell_12_readvariableop_6_resource:H??'gru_12/while/gru_cell_12/ReadVariableOp?)gru_12/while/gru_cell_12/ReadVariableOp_1?)gru_12/while/gru_cell_12/ReadVariableOp_2?)gru_12/while/gru_cell_12/ReadVariableOp_3?)gru_12/while/gru_cell_12/ReadVariableOp_4?)gru_12/while/gru_cell_12/ReadVariableOp_5?)gru_12/while/gru_cell_12/ReadVariableOp_6?)gru_12/while/gru_cell_12/ReadVariableOp_7?)gru_12/while/gru_cell_12/ReadVariableOp_8?
>gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>gru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0gru_12_while_placeholderGgru_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_12/while/TensorArrayV2Read/TensorListGetItem?
'gru_12/while/gru_cell_12/ReadVariableOpReadVariableOp2gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02)
'gru_12/while/gru_cell_12/ReadVariableOp?
,gru_12/while/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,gru_12/while/gru_cell_12/strided_slice/stack?
.gru_12/while/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.gru_12/while/gru_cell_12/strided_slice/stack_1?
.gru_12/while/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_12/while/gru_cell_12/strided_slice/stack_2?
&gru_12/while/gru_cell_12/strided_sliceStridedSlice/gru_12/while/gru_cell_12/ReadVariableOp:value:05gru_12/while/gru_cell_12/strided_slice/stack:output:07gru_12/while/gru_cell_12/strided_slice/stack_1:output:07gru_12/while/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2(
&gru_12/while/gru_cell_12/strided_slice?
gru_12/while/gru_cell_12/MatMulMatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_12/while/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2!
gru_12/while/gru_cell_12/MatMul?
)gru_12/while/gru_cell_12/ReadVariableOp_1ReadVariableOp2gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_1?
.gru_12/while/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.gru_12/while/gru_cell_12/strided_slice_1/stack?
0gru_12/while/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   22
0gru_12/while/gru_cell_12/strided_slice_1/stack_1?
0gru_12/while/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_1/stack_2?
(gru_12/while/gru_cell_12/strided_slice_1StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_1:value:07gru_12/while/gru_cell_12/strided_slice_1/stack:output:09gru_12/while/gru_cell_12/strided_slice_1/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_1?
!gru_12/while/gru_cell_12/MatMul_1MatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:01gru_12/while/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_1?
)gru_12/while/gru_cell_12/ReadVariableOp_2ReadVariableOp2gru_12_while_gru_cell_12_readvariableop_resource_0*
_output_shapes
:	?H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_2?
.gru_12/while/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   20
.gru_12/while/gru_cell_12/strided_slice_2/stack?
0gru_12/while/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0gru_12/while/gru_cell_12/strided_slice_2/stack_1?
0gru_12/while/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_2/stack_2?
(gru_12/while/gru_cell_12/strided_slice_2StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_2:value:07gru_12/while/gru_cell_12/strided_slice_2/stack:output:09gru_12/while/gru_cell_12/strided_slice_2/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_2?
!gru_12/while/gru_cell_12/MatMul_2MatMul7gru_12/while/TensorArrayV2Read/TensorListGetItem:item:01gru_12/while/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_2?
)gru_12/while/gru_cell_12/ReadVariableOp_3ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_3?
.gru_12/while/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.gru_12/while/gru_cell_12/strided_slice_3/stack?
0gru_12/while/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_3/stack_1?
0gru_12/while/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_3/stack_2?
(gru_12/while/gru_cell_12/strided_slice_3StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_3:value:07gru_12/while/gru_cell_12/strided_slice_3/stack:output:09gru_12/while/gru_cell_12/strided_slice_3/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2*
(gru_12/while/gru_cell_12/strided_slice_3?
 gru_12/while/gru_cell_12/BiasAddBiasAdd)gru_12/while/gru_cell_12/MatMul:product:01gru_12/while/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2"
 gru_12/while/gru_cell_12/BiasAdd?
)gru_12/while/gru_cell_12/ReadVariableOp_4ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_4?
.gru_12/while/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.gru_12/while/gru_cell_12/strided_slice_4/stack?
0gru_12/while/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:022
0gru_12/while/gru_cell_12/strided_slice_4/stack_1?
0gru_12/while/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_4/stack_2?
(gru_12/while/gru_cell_12/strided_slice_4StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_4:value:07gru_12/while/gru_cell_12/strided_slice_4/stack:output:09gru_12/while/gru_cell_12/strided_slice_4/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2*
(gru_12/while/gru_cell_12/strided_slice_4?
"gru_12/while/gru_cell_12/BiasAdd_1BiasAdd+gru_12/while/gru_cell_12/MatMul_1:product:01gru_12/while/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2$
"gru_12/while/gru_cell_12/BiasAdd_1?
)gru_12/while/gru_cell_12/ReadVariableOp_5ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_3_resource_0*
_output_shapes
:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_5?
.gru_12/while/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:020
.gru_12/while/gru_cell_12/strided_slice_5/stack?
0gru_12/while/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0gru_12/while/gru_cell_12/strided_slice_5/stack_1?
0gru_12/while/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0gru_12/while/gru_cell_12/strided_slice_5/stack_2?
(gru_12/while/gru_cell_12/strided_slice_5StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_5:value:07gru_12/while/gru_cell_12/strided_slice_5/stack:output:09gru_12/while/gru_cell_12/strided_slice_5/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_5?
"gru_12/while/gru_cell_12/BiasAdd_2BiasAdd+gru_12/while/gru_cell_12/MatMul_2:product:01gru_12/while/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2$
"gru_12/while/gru_cell_12/BiasAdd_2?
)gru_12/while/gru_cell_12/ReadVariableOp_6ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_6?
.gru_12/while/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.gru_12/while/gru_cell_12/strided_slice_6/stack?
0gru_12/while/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0gru_12/while/gru_cell_12/strided_slice_6/stack_1?
0gru_12/while/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_6/stack_2?
(gru_12/while/gru_cell_12/strided_slice_6StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_6:value:07gru_12/while/gru_cell_12/strided_slice_6/stack:output:09gru_12/while/gru_cell_12/strided_slice_6/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_6?
!gru_12/while/gru_cell_12/MatMul_3MatMulgru_12_while_placeholder_21gru_12/while/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_3?
)gru_12/while/gru_cell_12/ReadVariableOp_7ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_7?
.gru_12/while/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.gru_12/while/gru_cell_12/strided_slice_7/stack?
0gru_12/while/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   22
0gru_12/while/gru_cell_12/strided_slice_7/stack_1?
0gru_12/while/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_7/stack_2?
(gru_12/while/gru_cell_12/strided_slice_7StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_7:value:07gru_12/while/gru_cell_12/strided_slice_7/stack:output:09gru_12/while/gru_cell_12/strided_slice_7/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_7?
!gru_12/while/gru_cell_12/MatMul_4MatMulgru_12_while_placeholder_21gru_12/while/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_4?
gru_12/while/gru_cell_12/addAddV2)gru_12/while/gru_cell_12/BiasAdd:output:0+gru_12/while/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/add?
gru_12/while/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
gru_12/while/gru_cell_12/Const?
 gru_12/while/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 gru_12/while/gru_cell_12/Const_1?
gru_12/while/gru_cell_12/MulMul gru_12/while/gru_cell_12/add:z:0'gru_12/while/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/Mul?
gru_12/while/gru_cell_12/Add_1AddV2 gru_12/while/gru_cell_12/Mul:z:0)gru_12/while/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/Add_1?
0gru_12/while/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0gru_12/while/gru_cell_12/clip_by_value/Minimum/y?
.gru_12/while/gru_cell_12/clip_by_value/MinimumMinimum"gru_12/while/gru_cell_12/Add_1:z:09gru_12/while/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????20
.gru_12/while/gru_cell_12/clip_by_value/Minimum?
(gru_12/while/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(gru_12/while/gru_cell_12/clip_by_value/y?
&gru_12/while/gru_cell_12/clip_by_valueMaximum2gru_12/while/gru_cell_12/clip_by_value/Minimum:z:01gru_12/while/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2(
&gru_12/while/gru_cell_12/clip_by_value?
gru_12/while/gru_cell_12/add_2AddV2+gru_12/while/gru_cell_12/BiasAdd_1:output:0+gru_12/while/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/add_2?
 gru_12/while/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 gru_12/while/gru_cell_12/Const_2?
 gru_12/while/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 gru_12/while/gru_cell_12/Const_3?
gru_12/while/gru_cell_12/Mul_1Mul"gru_12/while/gru_cell_12/add_2:z:0)gru_12/while/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/Mul_1?
gru_12/while/gru_cell_12/Add_3AddV2"gru_12/while/gru_cell_12/Mul_1:z:0)gru_12/while/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/Add_3?
2gru_12/while/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2gru_12/while/gru_cell_12/clip_by_value_1/Minimum/y?
0gru_12/while/gru_cell_12/clip_by_value_1/MinimumMinimum"gru_12/while/gru_cell_12/Add_3:z:0;gru_12/while/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0gru_12/while/gru_cell_12/clip_by_value_1/Minimum?
*gru_12/while/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*gru_12/while/gru_cell_12/clip_by_value_1/y?
(gru_12/while/gru_cell_12/clip_by_value_1Maximum4gru_12/while/gru_cell_12/clip_by_value_1/Minimum:z:03gru_12/while/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2*
(gru_12/while/gru_cell_12/clip_by_value_1?
gru_12/while/gru_cell_12/mul_2Mul,gru_12/while/gru_cell_12/clip_by_value_1:z:0gru_12_while_placeholder_2*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/mul_2?
)gru_12/while/gru_cell_12/ReadVariableOp_8ReadVariableOp4gru_12_while_gru_cell_12_readvariableop_6_resource_0*
_output_shapes

:H*
dtype02+
)gru_12/while/gru_cell_12/ReadVariableOp_8?
.gru_12/while/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   20
.gru_12/while/gru_cell_12/strided_slice_8/stack?
0gru_12/while/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0gru_12/while/gru_cell_12/strided_slice_8/stack_1?
0gru_12/while/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0gru_12/while/gru_cell_12/strided_slice_8/stack_2?
(gru_12/while/gru_cell_12/strided_slice_8StridedSlice1gru_12/while/gru_cell_12/ReadVariableOp_8:value:07gru_12/while/gru_cell_12/strided_slice_8/stack:output:09gru_12/while/gru_cell_12/strided_slice_8/stack_1:output:09gru_12/while/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(gru_12/while/gru_cell_12/strided_slice_8?
!gru_12/while/gru_cell_12/MatMul_5MatMul"gru_12/while/gru_cell_12/mul_2:z:01gru_12/while/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2#
!gru_12/while/gru_cell_12/MatMul_5?
gru_12/while/gru_cell_12/add_4AddV2+gru_12/while/gru_cell_12/BiasAdd_2:output:0+gru_12/while/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/add_4?
gru_12/while/gru_cell_12/ReluRelu"gru_12/while/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/Relu?
gru_12/while/gru_cell_12/mul_3Mul*gru_12/while/gru_cell_12/clip_by_value:z:0gru_12_while_placeholder_2*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/mul_3?
gru_12/while/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_12/while/gru_cell_12/sub/x?
gru_12/while/gru_cell_12/subSub'gru_12/while/gru_cell_12/sub/x:output:0*gru_12/while/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_12/while/gru_cell_12/sub?
gru_12/while/gru_cell_12/mul_4Mul gru_12/while/gru_cell_12/sub:z:0+gru_12/while/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/mul_4?
gru_12/while/gru_cell_12/add_5AddV2"gru_12/while/gru_cell_12/mul_3:z:0"gru_12/while/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2 
gru_12/while/gru_cell_12/add_5?
1gru_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_12_while_placeholder_1gru_12_while_placeholder"gru_12/while/gru_cell_12/add_5:z:0*
_output_shapes
: *
element_dtype023
1gru_12/while/TensorArrayV2Write/TensorListSetItemj
gru_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_12/while/add/y?
gru_12/while/addAddV2gru_12_while_placeholdergru_12/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_12/while/addn
gru_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_12/while/add_1/y?
gru_12/while/add_1AddV2&gru_12_while_gru_12_while_loop_countergru_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_12/while/add_1?
gru_12/while/IdentityIdentitygru_12/while/add_1:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity?
gru_12/while/Identity_1Identity,gru_12_while_gru_12_while_maximum_iterations^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity_1?
gru_12/while/Identity_2Identitygru_12/while/add:z:0^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity_2?
gru_12/while/Identity_3IdentityAgru_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_12/while/NoOp*
T0*
_output_shapes
: 2
gru_12/while/Identity_3?
gru_12/while/Identity_4Identity"gru_12/while/gru_cell_12/add_5:z:0^gru_12/while/NoOp*
T0*'
_output_shapes
:?????????2
gru_12/while/Identity_4?
gru_12/while/NoOpNoOp(^gru_12/while/gru_cell_12/ReadVariableOp*^gru_12/while/gru_cell_12/ReadVariableOp_1*^gru_12/while/gru_cell_12/ReadVariableOp_2*^gru_12/while/gru_cell_12/ReadVariableOp_3*^gru_12/while/gru_cell_12/ReadVariableOp_4*^gru_12/while/gru_cell_12/ReadVariableOp_5*^gru_12/while/gru_cell_12/ReadVariableOp_6*^gru_12/while/gru_cell_12/ReadVariableOp_7*^gru_12/while/gru_cell_12/ReadVariableOp_8*"
_acd_function_control_output(*
_output_shapes
 2
gru_12/while/NoOp"L
#gru_12_while_gru_12_strided_slice_1%gru_12_while_gru_12_strided_slice_1_0"j
2gru_12_while_gru_cell_12_readvariableop_3_resource4gru_12_while_gru_cell_12_readvariableop_3_resource_0"j
2gru_12_while_gru_cell_12_readvariableop_6_resource4gru_12_while_gru_cell_12_readvariableop_6_resource_0"f
0gru_12_while_gru_cell_12_readvariableop_resource2gru_12_while_gru_cell_12_readvariableop_resource_0"7
gru_12_while_identitygru_12/while/Identity:output:0";
gru_12_while_identity_1 gru_12/while/Identity_1:output:0";
gru_12_while_identity_2 gru_12/while/Identity_2:output:0";
gru_12_while_identity_3 gru_12/while/Identity_3:output:0";
gru_12_while_identity_4 gru_12/while/Identity_4:output:0"?
_gru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensoragru_12_while_tensorarrayv2read_tensorlistgetitem_gru_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2R
'gru_12/while/gru_cell_12/ReadVariableOp'gru_12/while/gru_cell_12/ReadVariableOp2V
)gru_12/while/gru_cell_12/ReadVariableOp_1)gru_12/while/gru_cell_12/ReadVariableOp_12V
)gru_12/while/gru_cell_12/ReadVariableOp_2)gru_12/while/gru_cell_12/ReadVariableOp_22V
)gru_12/while/gru_cell_12/ReadVariableOp_3)gru_12/while/gru_cell_12/ReadVariableOp_32V
)gru_12/while/gru_cell_12/ReadVariableOp_4)gru_12/while/gru_cell_12/ReadVariableOp_42V
)gru_12/while/gru_cell_12/ReadVariableOp_5)gru_12/while/gru_cell_12/ReadVariableOp_52V
)gru_12/while/gru_cell_12/ReadVariableOp_6)gru_12/while/gru_cell_12/ReadVariableOp_62V
)gru_12/while/gru_cell_12/ReadVariableOp_7)gru_12/while/gru_cell_12/ReadVariableOp_72V
)gru_12/while/gru_cell_12/ReadVariableOp_8)gru_12/while/gru_cell_12/ReadVariableOp_8: 
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
while_cond_292267
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_292267___redundant_placeholder04
0while_while_cond_292267___redundant_placeholder14
0while_while_cond_292267___redundant_placeholder24
0while_while_cond_292267___redundant_placeholder3
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
b
F__inference_flatten_12_layer_call_and_return_conditional_losses_291568

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
?
?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_289546

inputs!
dense_24_289536: 
dense_24_289538: 
identity?? dense_24/StatefulPartitionedCallD
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
 dense_24/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_24_289536dense_24_289538*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2894872"
 dense_24/StatefulPartitionedCallq
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
	Reshape_1Reshape)dense_24/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_1z
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityq
NoOpNoOp!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
B__inference_gru_12_layer_call_and_return_conditional_losses_292406

inputs6
#gru_cell_12_readvariableop_resource:	?H3
%gru_cell_12_readvariableop_3_resource:H7
%gru_cell_12_readvariableop_6_resource:H
identity??gru_cell_12/ReadVariableOp?gru_cell_12/ReadVariableOp_1?gru_cell_12/ReadVariableOp_2?gru_cell_12/ReadVariableOp_3?gru_cell_12/ReadVariableOp_4?gru_cell_12/ReadVariableOp_5?gru_cell_12/ReadVariableOp_6?gru_cell_12/ReadVariableOp_7?gru_cell_12/ReadVariableOp_8?whileD
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
gru_cell_12/ReadVariableOpReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp?
gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
gru_cell_12/strided_slice/stack?
!gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice/stack_1?
!gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell_12/strided_slice/stack_2?
gru_cell_12/strided_sliceStridedSlice"gru_cell_12/ReadVariableOp:value:0(gru_cell_12/strided_slice/stack:output:0*gru_cell_12/strided_slice/stack_1:output:0*gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice?
gru_cell_12/MatMulMatMulstrided_slice_2:output:0"gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul?
gru_cell_12/ReadVariableOp_1ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_1?
!gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_1/stack?
#gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_1/stack_1?
#gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_1/stack_2?
gru_cell_12/strided_slice_1StridedSlice$gru_cell_12/ReadVariableOp_1:value:0*gru_cell_12/strided_slice_1/stack:output:0,gru_cell_12/strided_slice_1/stack_1:output:0,gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_1?
gru_cell_12/MatMul_1MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_1?
gru_cell_12/ReadVariableOp_2ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_2?
!gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_2/stack?
#gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_2/stack_1?
#gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_2/stack_2?
gru_cell_12/strided_slice_2StridedSlice$gru_cell_12/ReadVariableOp_2:value:0*gru_cell_12/strided_slice_2/stack:output:0,gru_cell_12/strided_slice_2/stack_1:output:0,gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_2?
gru_cell_12/MatMul_2MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_2?
gru_cell_12/ReadVariableOp_3ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_3?
!gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell_12/strided_slice_3/stack?
#gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_1?
#gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_2?
gru_cell_12/strided_slice_3StridedSlice$gru_cell_12/ReadVariableOp_3:value:0*gru_cell_12/strided_slice_3/stack:output:0,gru_cell_12/strided_slice_3/stack_1:output:0,gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell_12/strided_slice_3?
gru_cell_12/BiasAddBiasAddgru_cell_12/MatMul:product:0$gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd?
gru_cell_12/ReadVariableOp_4ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_4?
!gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell_12/strided_slice_4/stack?
#gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02%
#gru_cell_12/strided_slice_4/stack_1?
#gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_4/stack_2?
gru_cell_12/strided_slice_4StridedSlice$gru_cell_12/ReadVariableOp_4:value:0*gru_cell_12/strided_slice_4/stack:output:0,gru_cell_12/strided_slice_4/stack_1:output:0,gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell_12/strided_slice_4?
gru_cell_12/BiasAdd_1BiasAddgru_cell_12/MatMul_1:product:0$gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_1?
gru_cell_12/ReadVariableOp_5ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_5?
!gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02#
!gru_cell_12/strided_slice_5/stack?
#gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_12/strided_slice_5/stack_1?
#gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_5/stack_2?
gru_cell_12/strided_slice_5StridedSlice$gru_cell_12/ReadVariableOp_5:value:0*gru_cell_12/strided_slice_5/stack:output:0,gru_cell_12/strided_slice_5/stack_1:output:0,gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell_12/strided_slice_5?
gru_cell_12/BiasAdd_2BiasAddgru_cell_12/MatMul_2:product:0$gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_2?
gru_cell_12/ReadVariableOp_6ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_6?
!gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell_12/strided_slice_6/stack?
#gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#gru_cell_12/strided_slice_6/stack_1?
#gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_6/stack_2?
gru_cell_12/strided_slice_6StridedSlice$gru_cell_12/ReadVariableOp_6:value:0*gru_cell_12/strided_slice_6/stack:output:0,gru_cell_12/strided_slice_6/stack_1:output:0,gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_6?
gru_cell_12/MatMul_3MatMulzeros:output:0$gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_3?
gru_cell_12/ReadVariableOp_7ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_7?
!gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_7/stack?
#gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_7/stack_1?
#gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_7/stack_2?
gru_cell_12/strided_slice_7StridedSlice$gru_cell_12/ReadVariableOp_7:value:0*gru_cell_12/strided_slice_7/stack:output:0,gru_cell_12/strided_slice_7/stack_1:output:0,gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_7?
gru_cell_12/MatMul_4MatMulzeros:output:0$gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_4?
gru_cell_12/addAddV2gru_cell_12/BiasAdd:output:0gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/addk
gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Consto
gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_1?
gru_cell_12/MulMulgru_cell_12/add:z:0gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul?
gru_cell_12/Add_1AddV2gru_cell_12/Mul:z:0gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_1?
#gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#gru_cell_12/clip_by_value/Minimum/y?
!gru_cell_12/clip_by_value/MinimumMinimumgru_cell_12/Add_1:z:0,gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2#
!gru_cell_12/clip_by_value/Minimum
gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value/y?
gru_cell_12/clip_by_valueMaximum%gru_cell_12/clip_by_value/Minimum:z:0$gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value?
gru_cell_12/add_2AddV2gru_cell_12/BiasAdd_1:output:0gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_2o
gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Const_2o
gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_3?
gru_cell_12/Mul_1Mulgru_cell_12/add_2:z:0gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul_1?
gru_cell_12/Add_3AddV2gru_cell_12/Mul_1:z:0gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_3?
%gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gru_cell_12/clip_by_value_1/Minimum/y?
#gru_cell_12/clip_by_value_1/MinimumMinimumgru_cell_12/Add_3:z:0.gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2%
#gru_cell_12/clip_by_value_1/Minimum?
gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value_1/y?
gru_cell_12/clip_by_value_1Maximum'gru_cell_12/clip_by_value_1/Minimum:z:0&gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value_1?
gru_cell_12/mul_2Mulgru_cell_12/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_2?
gru_cell_12/ReadVariableOp_8ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_8?
!gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_8/stack?
#gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_8/stack_1?
#gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_8/stack_2?
gru_cell_12/strided_slice_8StridedSlice$gru_cell_12/ReadVariableOp_8:value:0*gru_cell_12/strided_slice_8/stack:output:0,gru_cell_12/strided_slice_8/stack_1:output:0,gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_8?
gru_cell_12/MatMul_5MatMulgru_cell_12/mul_2:z:0$gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_5?
gru_cell_12/add_4AddV2gru_cell_12/BiasAdd_2:output:0gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_4u
gru_cell_12/ReluRelugru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Relu?
gru_cell_12/mul_3Mulgru_cell_12/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_3k
gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_12/sub/x?
gru_cell_12/subSubgru_cell_12/sub/x:output:0gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/sub?
gru_cell_12/mul_4Mulgru_cell_12/sub:z:0gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_4?
gru_cell_12/add_5AddV2gru_cell_12/mul_3:z:0gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_5?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_12_readvariableop_resource%gru_cell_12_readvariableop_3_resource%gru_cell_12_readvariableop_6_resource*
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
bodyR
while_body_292268*
condR
while_cond_292267*8
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
NoOpNoOp^gru_cell_12/ReadVariableOp^gru_cell_12/ReadVariableOp_1^gru_cell_12/ReadVariableOp_2^gru_cell_12/ReadVariableOp_3^gru_cell_12/ReadVariableOp_4^gru_cell_12/ReadVariableOp_5^gru_cell_12/ReadVariableOp_6^gru_cell_12/ReadVariableOp_7^gru_cell_12/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
gru_cell_12/ReadVariableOpgru_cell_12/ReadVariableOp2<
gru_cell_12/ReadVariableOp_1gru_cell_12/ReadVariableOp_12<
gru_cell_12/ReadVariableOp_2gru_cell_12/ReadVariableOp_22<
gru_cell_12/ReadVariableOp_3gru_cell_12/ReadVariableOp_32<
gru_cell_12/ReadVariableOp_4gru_cell_12/ReadVariableOp_42<
gru_cell_12/ReadVariableOp_5gru_cell_12/ReadVariableOp_52<
gru_cell_12/ReadVariableOp_6gru_cell_12/ReadVariableOp_62<
gru_cell_12/ReadVariableOp_7gru_cell_12/ReadVariableOp_72<
gru_cell_12/ReadVariableOp_8gru_cell_12/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_12_layer_call_fn_290156
conv1d_24_input
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
StatefulPartitionedCallStatefulPartitionedCallconv1d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
GPU2*0J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_2901312
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
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameconv1d_24_input
?a
?
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_293107

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
?
?
E__inference_conv1d_24_layer_call_and_return_conditional_losses_289764

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
?
M
1__inference_max_pooling1d_12_layer_call_fn_291541

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
GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_2897992
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
?
?
)__inference_dense_24_layer_call_fn_293116

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
GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2894872
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
??
?
!__inference__wrapped_model_288729
conv1d_24_inputY
Csequential_12_conv1d_24_conv1d_expanddims_1_readvariableop_resource: E
7sequential_12_conv1d_24_biasadd_readvariableop_resource: Y
Csequential_12_conv1d_25_conv1d_expanddims_1_readvariableop_resource: @E
7sequential_12_conv1d_25_biasadd_readvariableop_resource:@K
8sequential_12_gru_12_gru_cell_12_readvariableop_resource:	?HH
:sequential_12_gru_12_gru_cell_12_readvariableop_3_resource:HL
:sequential_12_gru_12_gru_cell_12_readvariableop_6_resource:H[
Isequential_12_time_distributed_24_dense_24_matmul_readvariableop_resource: X
Jsequential_12_time_distributed_24_dense_24_biasadd_readvariableop_resource: [
Isequential_12_time_distributed_25_dense_25_matmul_readvariableop_resource: X
Jsequential_12_time_distributed_25_dense_25_biasadd_readvariableop_resource:
identity??.sequential_12/conv1d_24/BiasAdd/ReadVariableOp?:sequential_12/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?.sequential_12/conv1d_25/BiasAdd/ReadVariableOp?:sequential_12/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?/sequential_12/gru_12/gru_cell_12/ReadVariableOp?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_1?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_2?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_3?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_4?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_5?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_6?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_7?1sequential_12/gru_12/gru_cell_12/ReadVariableOp_8?sequential_12/gru_12/while?Asequential_12/time_distributed_24/dense_24/BiasAdd/ReadVariableOp?@sequential_12/time_distributed_24/dense_24/MatMul/ReadVariableOp?Asequential_12/time_distributed_25/dense_25/BiasAdd/ReadVariableOp?@sequential_12/time_distributed_25/dense_25/MatMul/ReadVariableOp?
-sequential_12/conv1d_24/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_12/conv1d_24/conv1d/ExpandDims/dim?
)sequential_12/conv1d_24/conv1d/ExpandDims
ExpandDimsconv1d_24_input6sequential_12/conv1d_24/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2+
)sequential_12/conv1d_24/conv1d/ExpandDims?
:sequential_12/conv1d_24/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_12_conv1d_24_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02<
:sequential_12/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp?
/sequential_12/conv1d_24/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_12/conv1d_24/conv1d/ExpandDims_1/dim?
+sequential_12/conv1d_24/conv1d/ExpandDims_1
ExpandDimsBsequential_12/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_12/conv1d_24/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2-
+sequential_12/conv1d_24/conv1d/ExpandDims_1?
sequential_12/conv1d_24/conv1dConv2D2sequential_12/conv1d_24/conv1d/ExpandDims:output:04sequential_12/conv1d_24/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2 
sequential_12/conv1d_24/conv1d?
&sequential_12/conv1d_24/conv1d/SqueezeSqueeze'sequential_12/conv1d_24/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2(
&sequential_12/conv1d_24/conv1d/Squeeze?
.sequential_12/conv1d_24/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv1d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_12/conv1d_24/BiasAdd/ReadVariableOp?
sequential_12/conv1d_24/BiasAddBiasAdd/sequential_12/conv1d_24/conv1d/Squeeze:output:06sequential_12/conv1d_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2!
sequential_12/conv1d_24/BiasAdd?
sequential_12/conv1d_24/ReluRelu(sequential_12/conv1d_24/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
sequential_12/conv1d_24/Relu?
-sequential_12/conv1d_25/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_12/conv1d_25/conv1d/ExpandDims/dim?
)sequential_12/conv1d_25/conv1d/ExpandDims
ExpandDims*sequential_12/conv1d_24/Relu:activations:06sequential_12/conv1d_25/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2+
)sequential_12/conv1d_25/conv1d/ExpandDims?
:sequential_12/conv1d_25/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_12_conv1d_25_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02<
:sequential_12/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp?
/sequential_12/conv1d_25/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_12/conv1d_25/conv1d/ExpandDims_1/dim?
+sequential_12/conv1d_25/conv1d/ExpandDims_1
ExpandDimsBsequential_12/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_12/conv1d_25/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2-
+sequential_12/conv1d_25/conv1d/ExpandDims_1?
sequential_12/conv1d_25/conv1dConv2D2sequential_12/conv1d_25/conv1d/ExpandDims:output:04sequential_12/conv1d_25/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2 
sequential_12/conv1d_25/conv1d?
&sequential_12/conv1d_25/conv1d/SqueezeSqueeze'sequential_12/conv1d_25/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2(
&sequential_12/conv1d_25/conv1d/Squeeze?
.sequential_12/conv1d_25/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv1d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_12/conv1d_25/BiasAdd/ReadVariableOp?
sequential_12/conv1d_25/BiasAddBiasAdd/sequential_12/conv1d_25/conv1d/Squeeze:output:06sequential_12/conv1d_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2!
sequential_12/conv1d_25/BiasAdd?
sequential_12/conv1d_25/ReluRelu(sequential_12/conv1d_25/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
sequential_12/conv1d_25/Relu?
-sequential_12/max_pooling1d_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_12/max_pooling1d_12/ExpandDims/dim?
)sequential_12/max_pooling1d_12/ExpandDims
ExpandDims*sequential_12/conv1d_25/Relu:activations:06sequential_12/max_pooling1d_12/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@2+
)sequential_12/max_pooling1d_12/ExpandDims?
&sequential_12/max_pooling1d_12/MaxPoolMaxPool2sequential_12/max_pooling1d_12/ExpandDims:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling1d_12/MaxPool?
&sequential_12/max_pooling1d_12/SqueezeSqueeze/sequential_12/max_pooling1d_12/MaxPool:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims
2(
&sequential_12/max_pooling1d_12/Squeeze?
sequential_12/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2 
sequential_12/flatten_12/Const?
 sequential_12/flatten_12/ReshapeReshape/sequential_12/max_pooling1d_12/Squeeze:output:0'sequential_12/flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_12/flatten_12/Reshape?
-sequential_12/repeat_vector_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_12/repeat_vector_12/ExpandDims/dim?
)sequential_12/repeat_vector_12/ExpandDims
ExpandDims)sequential_12/flatten_12/Reshape:output:06sequential_12/repeat_vector_12/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2+
)sequential_12/repeat_vector_12/ExpandDims?
$sequential_12/repeat_vector_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2&
$sequential_12/repeat_vector_12/stack?
#sequential_12/repeat_vector_12/TileTile2sequential_12/repeat_vector_12/ExpandDims:output:0-sequential_12/repeat_vector_12/stack:output:0*
T0*,
_output_shapes
:??????????2%
#sequential_12/repeat_vector_12/Tile?
sequential_12/gru_12/ShapeShape,sequential_12/repeat_vector_12/Tile:output:0*
T0*
_output_shapes
:2
sequential_12/gru_12/Shape?
(sequential_12/gru_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/gru_12/strided_slice/stack?
*sequential_12/gru_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_12/gru_12/strided_slice/stack_1?
*sequential_12/gru_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_12/gru_12/strided_slice/stack_2?
"sequential_12/gru_12/strided_sliceStridedSlice#sequential_12/gru_12/Shape:output:01sequential_12/gru_12/strided_slice/stack:output:03sequential_12/gru_12/strided_slice/stack_1:output:03sequential_12/gru_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_12/gru_12/strided_slice?
 sequential_12/gru_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_12/gru_12/zeros/mul/y?
sequential_12/gru_12/zeros/mulMul+sequential_12/gru_12/strided_slice:output:0)sequential_12/gru_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_12/gru_12/zeros/mul?
!sequential_12/gru_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_12/gru_12/zeros/Less/y?
sequential_12/gru_12/zeros/LessLess"sequential_12/gru_12/zeros/mul:z:0*sequential_12/gru_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/gru_12/zeros/Less?
#sequential_12/gru_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/gru_12/zeros/packed/1?
!sequential_12/gru_12/zeros/packedPack+sequential_12/gru_12/strided_slice:output:0,sequential_12/gru_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_12/gru_12/zeros/packed?
 sequential_12/gru_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_12/gru_12/zeros/Const?
sequential_12/gru_12/zerosFill*sequential_12/gru_12/zeros/packed:output:0)sequential_12/gru_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_12/gru_12/zeros?
#sequential_12/gru_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_12/gru_12/transpose/perm?
sequential_12/gru_12/transpose	Transpose,sequential_12/repeat_vector_12/Tile:output:0,sequential_12/gru_12/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2 
sequential_12/gru_12/transpose?
sequential_12/gru_12/Shape_1Shape"sequential_12/gru_12/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/gru_12/Shape_1?
*sequential_12/gru_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_12/gru_12/strided_slice_1/stack?
,sequential_12/gru_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/gru_12/strided_slice_1/stack_1?
,sequential_12/gru_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/gru_12/strided_slice_1/stack_2?
$sequential_12/gru_12/strided_slice_1StridedSlice%sequential_12/gru_12/Shape_1:output:03sequential_12/gru_12/strided_slice_1/stack:output:05sequential_12/gru_12/strided_slice_1/stack_1:output:05sequential_12/gru_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_12/gru_12/strided_slice_1?
0sequential_12/gru_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_12/gru_12/TensorArrayV2/element_shape?
"sequential_12/gru_12/TensorArrayV2TensorListReserve9sequential_12/gru_12/TensorArrayV2/element_shape:output:0-sequential_12/gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_12/gru_12/TensorArrayV2?
Jsequential_12/gru_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2L
Jsequential_12/gru_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_12/gru_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_12/gru_12/transpose:y:0Ssequential_12/gru_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_12/gru_12/TensorArrayUnstack/TensorListFromTensor?
*sequential_12/gru_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_12/gru_12/strided_slice_2/stack?
,sequential_12/gru_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/gru_12/strided_slice_2/stack_1?
,sequential_12/gru_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/gru_12/strided_slice_2/stack_2?
$sequential_12/gru_12/strided_slice_2StridedSlice"sequential_12/gru_12/transpose:y:03sequential_12/gru_12/strided_slice_2/stack:output:05sequential_12/gru_12/strided_slice_2/stack_1:output:05sequential_12/gru_12/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_12/gru_12/strided_slice_2?
/sequential_12/gru_12/gru_cell_12/ReadVariableOpReadVariableOp8sequential_12_gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype021
/sequential_12/gru_12/gru_cell_12/ReadVariableOp?
4sequential_12/gru_12/gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4sequential_12/gru_12/gru_cell_12/strided_slice/stack?
6sequential_12/gru_12/gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_12/gru_12/gru_cell_12/strided_slice/stack_1?
6sequential_12/gru_12/gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6sequential_12/gru_12/gru_cell_12/strided_slice/stack_2?
.sequential_12/gru_12/gru_cell_12/strided_sliceStridedSlice7sequential_12/gru_12/gru_cell_12/ReadVariableOp:value:0=sequential_12/gru_12/gru_cell_12/strided_slice/stack:output:0?sequential_12/gru_12/gru_cell_12/strided_slice/stack_1:output:0?sequential_12/gru_12/gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask20
.sequential_12/gru_12/gru_cell_12/strided_slice?
'sequential_12/gru_12/gru_cell_12/MatMulMatMul-sequential_12/gru_12/strided_slice_2:output:07sequential_12/gru_12/gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2)
'sequential_12/gru_12/gru_cell_12/MatMul?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_1ReadVariableOp8sequential_12_gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_1?
6sequential_12/gru_12/gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_12/gru_12/gru_cell_12/strided_slice_1/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2:
8sequential_12/gru_12/gru_cell_12/strided_slice_1/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/gru_12/gru_cell_12/strided_slice_1/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_1StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_1:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_1/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_1/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask22
0sequential_12/gru_12/gru_cell_12/strided_slice_1?
)sequential_12/gru_12/gru_cell_12/MatMul_1MatMul-sequential_12/gru_12/strided_slice_2:output:09sequential_12/gru_12/gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_12/gru_12/gru_cell_12/MatMul_1?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_2ReadVariableOp8sequential_12_gru_12_gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_2?
6sequential_12/gru_12/gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   28
6sequential_12/gru_12/gru_cell_12/strided_slice_2/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_12/gru_12/gru_cell_12/strided_slice_2/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/gru_12/gru_cell_12/strided_slice_2/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_2StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_2:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_2/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_2/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask22
0sequential_12/gru_12/gru_cell_12/strided_slice_2?
)sequential_12/gru_12/gru_cell_12/MatMul_2MatMul-sequential_12/gru_12/strided_slice_2:output:09sequential_12/gru_12/gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_12/gru_12/gru_cell_12/MatMul_2?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_3ReadVariableOp:sequential_12_gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_3?
6sequential_12/gru_12/gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_12/gru_12/gru_cell_12/strided_slice_3/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_12/gru_12/gru_cell_12/strided_slice_3/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_12/gru_12/gru_cell_12/strided_slice_3/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_3StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_3:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_3/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_3/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask22
0sequential_12/gru_12/gru_cell_12/strided_slice_3?
(sequential_12/gru_12/gru_cell_12/BiasAddBiasAdd1sequential_12/gru_12/gru_cell_12/MatMul:product:09sequential_12/gru_12/gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2*
(sequential_12/gru_12/gru_cell_12/BiasAdd?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_4ReadVariableOp:sequential_12_gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_4?
6sequential_12/gru_12/gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6sequential_12/gru_12/gru_cell_12/strided_slice_4/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02:
8sequential_12/gru_12/gru_cell_12/strided_slice_4/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_12/gru_12/gru_cell_12/strided_slice_4/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_4StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_4:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_4/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_4/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:22
0sequential_12/gru_12/gru_cell_12/strided_slice_4?
*sequential_12/gru_12/gru_cell_12/BiasAdd_1BiasAdd3sequential_12/gru_12/gru_cell_12/MatMul_1:product:09sequential_12/gru_12/gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2,
*sequential_12/gru_12/gru_cell_12/BiasAdd_1?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_5ReadVariableOp:sequential_12_gru_12_gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_5?
6sequential_12/gru_12/gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:028
6sequential_12/gru_12/gru_cell_12/strided_slice_5/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_12/gru_12/gru_cell_12/strided_slice_5/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_12/gru_12/gru_cell_12/strided_slice_5/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_5StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_5:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_5/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_5/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask22
0sequential_12/gru_12/gru_cell_12/strided_slice_5?
*sequential_12/gru_12/gru_cell_12/BiasAdd_2BiasAdd3sequential_12/gru_12/gru_cell_12/MatMul_2:product:09sequential_12/gru_12/gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2,
*sequential_12/gru_12/gru_cell_12/BiasAdd_2?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_6ReadVariableOp:sequential_12_gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_6?
6sequential_12/gru_12/gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_12/gru_12/gru_cell_12/strided_slice_6/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2:
8sequential_12/gru_12/gru_cell_12/strided_slice_6/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/gru_12/gru_cell_12/strided_slice_6/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_6StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_6:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_6/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_6/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask22
0sequential_12/gru_12/gru_cell_12/strided_slice_6?
)sequential_12/gru_12/gru_cell_12/MatMul_3MatMul#sequential_12/gru_12/zeros:output:09sequential_12/gru_12/gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_12/gru_12/gru_cell_12/MatMul_3?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_7ReadVariableOp:sequential_12_gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_7?
6sequential_12/gru_12/gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_12/gru_12/gru_cell_12/strided_slice_7/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2:
8sequential_12/gru_12/gru_cell_12/strided_slice_7/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/gru_12/gru_cell_12/strided_slice_7/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_7StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_7:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_7/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_7/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask22
0sequential_12/gru_12/gru_cell_12/strided_slice_7?
)sequential_12/gru_12/gru_cell_12/MatMul_4MatMul#sequential_12/gru_12/zeros:output:09sequential_12/gru_12/gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_12/gru_12/gru_cell_12/MatMul_4?
$sequential_12/gru_12/gru_cell_12/addAddV21sequential_12/gru_12/gru_cell_12/BiasAdd:output:03sequential_12/gru_12/gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2&
$sequential_12/gru_12/gru_cell_12/add?
&sequential_12/gru_12/gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2(
&sequential_12/gru_12/gru_cell_12/Const?
(sequential_12/gru_12/gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(sequential_12/gru_12/gru_cell_12/Const_1?
$sequential_12/gru_12/gru_cell_12/MulMul(sequential_12/gru_12/gru_cell_12/add:z:0/sequential_12/gru_12/gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2&
$sequential_12/gru_12/gru_cell_12/Mul?
&sequential_12/gru_12/gru_cell_12/Add_1AddV2(sequential_12/gru_12/gru_cell_12/Mul:z:01sequential_12/gru_12/gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/Add_1?
8sequential_12/gru_12/gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2:
8sequential_12/gru_12/gru_cell_12/clip_by_value/Minimum/y?
6sequential_12/gru_12/gru_cell_12/clip_by_value/MinimumMinimum*sequential_12/gru_12/gru_cell_12/Add_1:z:0Asequential_12/gru_12/gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????28
6sequential_12/gru_12/gru_cell_12/clip_by_value/Minimum?
0sequential_12/gru_12/gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0sequential_12/gru_12/gru_cell_12/clip_by_value/y?
.sequential_12/gru_12/gru_cell_12/clip_by_valueMaximum:sequential_12/gru_12/gru_cell_12/clip_by_value/Minimum:z:09sequential_12/gru_12/gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????20
.sequential_12/gru_12/gru_cell_12/clip_by_value?
&sequential_12/gru_12/gru_cell_12/add_2AddV23sequential_12/gru_12/gru_cell_12/BiasAdd_1:output:03sequential_12/gru_12/gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/add_2?
(sequential_12/gru_12/gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2*
(sequential_12/gru_12/gru_cell_12/Const_2?
(sequential_12/gru_12/gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(sequential_12/gru_12/gru_cell_12/Const_3?
&sequential_12/gru_12/gru_cell_12/Mul_1Mul*sequential_12/gru_12/gru_cell_12/add_2:z:01sequential_12/gru_12/gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/Mul_1?
&sequential_12/gru_12/gru_cell_12/Add_3AddV2*sequential_12/gru_12/gru_cell_12/Mul_1:z:01sequential_12/gru_12/gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/Add_3?
:sequential_12/gru_12/gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2<
:sequential_12/gru_12/gru_cell_12/clip_by_value_1/Minimum/y?
8sequential_12/gru_12/gru_cell_12/clip_by_value_1/MinimumMinimum*sequential_12/gru_12/gru_cell_12/Add_3:z:0Csequential_12/gru_12/gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2:
8sequential_12/gru_12/gru_cell_12/clip_by_value_1/Minimum?
2sequential_12/gru_12/gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    24
2sequential_12/gru_12/gru_cell_12/clip_by_value_1/y?
0sequential_12/gru_12/gru_cell_12/clip_by_value_1Maximum<sequential_12/gru_12/gru_cell_12/clip_by_value_1/Minimum:z:0;sequential_12/gru_12/gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????22
0sequential_12/gru_12/gru_cell_12/clip_by_value_1?
&sequential_12/gru_12/gru_cell_12/mul_2Mul4sequential_12/gru_12/gru_cell_12/clip_by_value_1:z:0#sequential_12/gru_12/zeros:output:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/mul_2?
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_8ReadVariableOp:sequential_12_gru_12_gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype023
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_8?
6sequential_12/gru_12/gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   28
6sequential_12/gru_12/gru_cell_12/strided_slice_8/stack?
8sequential_12/gru_12/gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_12/gru_12/gru_cell_12/strided_slice_8/stack_1?
8sequential_12/gru_12/gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/gru_12/gru_cell_12/strided_slice_8/stack_2?
0sequential_12/gru_12/gru_cell_12/strided_slice_8StridedSlice9sequential_12/gru_12/gru_cell_12/ReadVariableOp_8:value:0?sequential_12/gru_12/gru_cell_12/strided_slice_8/stack:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_8/stack_1:output:0Asequential_12/gru_12/gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask22
0sequential_12/gru_12/gru_cell_12/strided_slice_8?
)sequential_12/gru_12/gru_cell_12/MatMul_5MatMul*sequential_12/gru_12/gru_cell_12/mul_2:z:09sequential_12/gru_12/gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_12/gru_12/gru_cell_12/MatMul_5?
&sequential_12/gru_12/gru_cell_12/add_4AddV23sequential_12/gru_12/gru_cell_12/BiasAdd_2:output:03sequential_12/gru_12/gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/add_4?
%sequential_12/gru_12/gru_cell_12/ReluRelu*sequential_12/gru_12/gru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2'
%sequential_12/gru_12/gru_cell_12/Relu?
&sequential_12/gru_12/gru_cell_12/mul_3Mul2sequential_12/gru_12/gru_cell_12/clip_by_value:z:0#sequential_12/gru_12/zeros:output:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/mul_3?
&sequential_12/gru_12/gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_12/gru_12/gru_cell_12/sub/x?
$sequential_12/gru_12/gru_cell_12/subSub/sequential_12/gru_12/gru_cell_12/sub/x:output:02sequential_12/gru_12/gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2&
$sequential_12/gru_12/gru_cell_12/sub?
&sequential_12/gru_12/gru_cell_12/mul_4Mul(sequential_12/gru_12/gru_cell_12/sub:z:03sequential_12/gru_12/gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/mul_4?
&sequential_12/gru_12/gru_cell_12/add_5AddV2*sequential_12/gru_12/gru_cell_12/mul_3:z:0*sequential_12/gru_12/gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2(
&sequential_12/gru_12/gru_cell_12/add_5?
2sequential_12/gru_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   24
2sequential_12/gru_12/TensorArrayV2_1/element_shape?
$sequential_12/gru_12/TensorArrayV2_1TensorListReserve;sequential_12/gru_12/TensorArrayV2_1/element_shape:output:0-sequential_12/gru_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_12/gru_12/TensorArrayV2_1x
sequential_12/gru_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/gru_12/time?
-sequential_12/gru_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_12/gru_12/while/maximum_iterations?
'sequential_12/gru_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_12/gru_12/while/loop_counter?
sequential_12/gru_12/whileWhile0sequential_12/gru_12/while/loop_counter:output:06sequential_12/gru_12/while/maximum_iterations:output:0"sequential_12/gru_12/time:output:0-sequential_12/gru_12/TensorArrayV2_1:handle:0#sequential_12/gru_12/zeros:output:0-sequential_12/gru_12/strided_slice_1:output:0Lsequential_12/gru_12/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_12_gru_12_gru_cell_12_readvariableop_resource:sequential_12_gru_12_gru_cell_12_readvariableop_3_resource:sequential_12_gru_12_gru_cell_12_readvariableop_6_resource*
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
_stateful_parallelism( *2
body*R(
&sequential_12_gru_12_while_body_288566*2
cond*R(
&sequential_12_gru_12_while_cond_288565*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
sequential_12/gru_12/while?
Esequential_12/gru_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esequential_12/gru_12/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_12/gru_12/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_12/gru_12/while:output:3Nsequential_12/gru_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype029
7sequential_12/gru_12/TensorArrayV2Stack/TensorListStack?
*sequential_12/gru_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_12/gru_12/strided_slice_3/stack?
,sequential_12/gru_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_12/gru_12/strided_slice_3/stack_1?
,sequential_12/gru_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/gru_12/strided_slice_3/stack_2?
$sequential_12/gru_12/strided_slice_3StridedSlice@sequential_12/gru_12/TensorArrayV2Stack/TensorListStack:tensor:03sequential_12/gru_12/strided_slice_3/stack:output:05sequential_12/gru_12/strided_slice_3/stack_1:output:05sequential_12/gru_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$sequential_12/gru_12/strided_slice_3?
%sequential_12/gru_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_12/gru_12/transpose_1/perm?
 sequential_12/gru_12/transpose_1	Transpose@sequential_12/gru_12/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_12/gru_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2"
 sequential_12/gru_12/transpose_1?
!sequential_12/dropout_12/IdentityIdentity$sequential_12/gru_12/transpose_1:y:0*
T0*+
_output_shapes
:?????????2#
!sequential_12/dropout_12/Identity?
/sequential_12/time_distributed_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   21
/sequential_12/time_distributed_24/Reshape/shape?
)sequential_12/time_distributed_24/ReshapeReshape*sequential_12/dropout_12/Identity:output:08sequential_12/time_distributed_24/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_12/time_distributed_24/Reshape?
@sequential_12/time_distributed_24/dense_24/MatMul/ReadVariableOpReadVariableOpIsequential_12_time_distributed_24_dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02B
@sequential_12/time_distributed_24/dense_24/MatMul/ReadVariableOp?
1sequential_12/time_distributed_24/dense_24/MatMulMatMul2sequential_12/time_distributed_24/Reshape:output:0Hsequential_12/time_distributed_24/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 23
1sequential_12/time_distributed_24/dense_24/MatMul?
Asequential_12/time_distributed_24/dense_24/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_time_distributed_24_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Asequential_12/time_distributed_24/dense_24/BiasAdd/ReadVariableOp?
2sequential_12/time_distributed_24/dense_24/BiasAddBiasAdd;sequential_12/time_distributed_24/dense_24/MatMul:product:0Isequential_12/time_distributed_24/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 24
2sequential_12/time_distributed_24/dense_24/BiasAdd?
1sequential_12/time_distributed_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       23
1sequential_12/time_distributed_24/Reshape_1/shape?
+sequential_12/time_distributed_24/Reshape_1Reshape;sequential_12/time_distributed_24/dense_24/BiasAdd:output:0:sequential_12/time_distributed_24/Reshape_1/shape:output:0*
T0*+
_output_shapes
:????????? 2-
+sequential_12/time_distributed_24/Reshape_1?
1sequential_12/time_distributed_24/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1sequential_12/time_distributed_24/Reshape_2/shape?
+sequential_12/time_distributed_24/Reshape_2Reshape*sequential_12/dropout_12/Identity:output:0:sequential_12/time_distributed_24/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_12/time_distributed_24/Reshape_2?
/sequential_12/time_distributed_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    21
/sequential_12/time_distributed_25/Reshape/shape?
)sequential_12/time_distributed_25/ReshapeReshape4sequential_12/time_distributed_24/Reshape_1:output:08sequential_12/time_distributed_25/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_12/time_distributed_25/Reshape?
@sequential_12/time_distributed_25/dense_25/MatMul/ReadVariableOpReadVariableOpIsequential_12_time_distributed_25_dense_25_matmul_readvariableop_resource*
_output_shapes

: *
dtype02B
@sequential_12/time_distributed_25/dense_25/MatMul/ReadVariableOp?
1sequential_12/time_distributed_25/dense_25/MatMulMatMul2sequential_12/time_distributed_25/Reshape:output:0Hsequential_12/time_distributed_25/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????23
1sequential_12/time_distributed_25/dense_25/MatMul?
Asequential_12/time_distributed_25/dense_25/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_time_distributed_25_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_12/time_distributed_25/dense_25/BiasAdd/ReadVariableOp?
2sequential_12/time_distributed_25/dense_25/BiasAddBiasAdd;sequential_12/time_distributed_25/dense_25/MatMul:product:0Isequential_12/time_distributed_25/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2sequential_12/time_distributed_25/dense_25/BiasAdd?
1sequential_12/time_distributed_25/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      23
1sequential_12/time_distributed_25/Reshape_1/shape?
+sequential_12/time_distributed_25/Reshape_1Reshape;sequential_12/time_distributed_25/dense_25/BiasAdd:output:0:sequential_12/time_distributed_25/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2-
+sequential_12/time_distributed_25/Reshape_1?
1sequential_12/time_distributed_25/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    23
1sequential_12/time_distributed_25/Reshape_2/shape?
+sequential_12/time_distributed_25/Reshape_2Reshape4sequential_12/time_distributed_24/Reshape_1:output:0:sequential_12/time_distributed_25/Reshape_2/shape:output:0*
T0*'
_output_shapes
:????????? 2-
+sequential_12/time_distributed_25/Reshape_2?
IdentityIdentity4sequential_12/time_distributed_25/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp/^sequential_12/conv1d_24/BiasAdd/ReadVariableOp;^sequential_12/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp/^sequential_12/conv1d_25/BiasAdd/ReadVariableOp;^sequential_12/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp0^sequential_12/gru_12/gru_cell_12/ReadVariableOp2^sequential_12/gru_12/gru_cell_12/ReadVariableOp_12^sequential_12/gru_12/gru_cell_12/ReadVariableOp_22^sequential_12/gru_12/gru_cell_12/ReadVariableOp_32^sequential_12/gru_12/gru_cell_12/ReadVariableOp_42^sequential_12/gru_12/gru_cell_12/ReadVariableOp_52^sequential_12/gru_12/gru_cell_12/ReadVariableOp_62^sequential_12/gru_12/gru_cell_12/ReadVariableOp_72^sequential_12/gru_12/gru_cell_12/ReadVariableOp_8^sequential_12/gru_12/whileB^sequential_12/time_distributed_24/dense_24/BiasAdd/ReadVariableOpA^sequential_12/time_distributed_24/dense_24/MatMul/ReadVariableOpB^sequential_12/time_distributed_25/dense_25/BiasAdd/ReadVariableOpA^sequential_12/time_distributed_25/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????: : : : : : : : : : : 2`
.sequential_12/conv1d_24/BiasAdd/ReadVariableOp.sequential_12/conv1d_24/BiasAdd/ReadVariableOp2x
:sequential_12/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp:sequential_12/conv1d_24/conv1d/ExpandDims_1/ReadVariableOp2`
.sequential_12/conv1d_25/BiasAdd/ReadVariableOp.sequential_12/conv1d_25/BiasAdd/ReadVariableOp2x
:sequential_12/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp:sequential_12/conv1d_25/conv1d/ExpandDims_1/ReadVariableOp2b
/sequential_12/gru_12/gru_cell_12/ReadVariableOp/sequential_12/gru_12/gru_cell_12/ReadVariableOp2f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_11sequential_12/gru_12/gru_cell_12/ReadVariableOp_12f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_21sequential_12/gru_12/gru_cell_12/ReadVariableOp_22f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_31sequential_12/gru_12/gru_cell_12/ReadVariableOp_32f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_41sequential_12/gru_12/gru_cell_12/ReadVariableOp_42f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_51sequential_12/gru_12/gru_cell_12/ReadVariableOp_52f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_61sequential_12/gru_12/gru_cell_12/ReadVariableOp_62f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_71sequential_12/gru_12/gru_cell_12/ReadVariableOp_72f
1sequential_12/gru_12/gru_cell_12/ReadVariableOp_81sequential_12/gru_12/gru_cell_12/ReadVariableOp_828
sequential_12/gru_12/whilesequential_12/gru_12/while2?
Asequential_12/time_distributed_24/dense_24/BiasAdd/ReadVariableOpAsequential_12/time_distributed_24/dense_24/BiasAdd/ReadVariableOp2?
@sequential_12/time_distributed_24/dense_24/MatMul/ReadVariableOp@sequential_12/time_distributed_24/dense_24/MatMul/ReadVariableOp2?
Asequential_12/time_distributed_25/dense_25/BiasAdd/ReadVariableOpAsequential_12/time_distributed_25/dense_25/BiasAdd/ReadVariableOp2?
@sequential_12/time_distributed_25/dense_25/MatMul/ReadVariableOp@sequential_12/time_distributed_25/dense_25/MatMul/ReadVariableOp:\ X
+
_output_shapes
:?????????
)
_user_specified_nameconv1d_24_input
?=
?
B__inference_gru_12_layer_call_and_return_conditional_losses_289231

inputs%
gru_cell_12_289156:	?H 
gru_cell_12_289158:H$
gru_cell_12_289160:H
identity??#gru_cell_12/StatefulPartitionedCall?whileD
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
#gru_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_12_289156gru_cell_12_289158gru_cell_12_289160*
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
GPU2*0J 8? *P
fKRI
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_2891012%
#gru_cell_12/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_12_289156gru_cell_12_289158gru_cell_12_289160*
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
bodyR
while_body_289168*
condR
while_cond_289167*8
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

Identity|
NoOpNoOp$^gru_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_12/StatefulPartitionedCall#gru_cell_12/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_291755
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_291755___redundant_placeholder04
0while_while_cond_291755___redundant_placeholder14
0while_while_cond_291755___redundant_placeholder24
0while_while_cond_291755___redundant_placeholder3
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
M
1__inference_repeat_vector_12_layer_call_fn_291573

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
GPU2*0J 8? *U
fPRN
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_2887692
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
?
?
E__inference_conv1d_24_layer_call_and_return_conditional_losses_291506

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
??
?
B__inference_gru_12_layer_call_and_return_conditional_losses_290073

inputs6
#gru_cell_12_readvariableop_resource:	?H3
%gru_cell_12_readvariableop_3_resource:H7
%gru_cell_12_readvariableop_6_resource:H
identity??gru_cell_12/ReadVariableOp?gru_cell_12/ReadVariableOp_1?gru_cell_12/ReadVariableOp_2?gru_cell_12/ReadVariableOp_3?gru_cell_12/ReadVariableOp_4?gru_cell_12/ReadVariableOp_5?gru_cell_12/ReadVariableOp_6?gru_cell_12/ReadVariableOp_7?gru_cell_12/ReadVariableOp_8?whileD
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
gru_cell_12/ReadVariableOpReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp?
gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
gru_cell_12/strided_slice/stack?
!gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice/stack_1?
!gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell_12/strided_slice/stack_2?
gru_cell_12/strided_sliceStridedSlice"gru_cell_12/ReadVariableOp:value:0(gru_cell_12/strided_slice/stack:output:0*gru_cell_12/strided_slice/stack_1:output:0*gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice?
gru_cell_12/MatMulMatMulstrided_slice_2:output:0"gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul?
gru_cell_12/ReadVariableOp_1ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_1?
!gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_1/stack?
#gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_1/stack_1?
#gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_1/stack_2?
gru_cell_12/strided_slice_1StridedSlice$gru_cell_12/ReadVariableOp_1:value:0*gru_cell_12/strided_slice_1/stack:output:0,gru_cell_12/strided_slice_1/stack_1:output:0,gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_1?
gru_cell_12/MatMul_1MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_1?
gru_cell_12/ReadVariableOp_2ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_2?
!gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_2/stack?
#gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_2/stack_1?
#gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_2/stack_2?
gru_cell_12/strided_slice_2StridedSlice$gru_cell_12/ReadVariableOp_2:value:0*gru_cell_12/strided_slice_2/stack:output:0,gru_cell_12/strided_slice_2/stack_1:output:0,gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_2?
gru_cell_12/MatMul_2MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_2?
gru_cell_12/ReadVariableOp_3ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_3?
!gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell_12/strided_slice_3/stack?
#gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_1?
#gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_2?
gru_cell_12/strided_slice_3StridedSlice$gru_cell_12/ReadVariableOp_3:value:0*gru_cell_12/strided_slice_3/stack:output:0,gru_cell_12/strided_slice_3/stack_1:output:0,gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell_12/strided_slice_3?
gru_cell_12/BiasAddBiasAddgru_cell_12/MatMul:product:0$gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd?
gru_cell_12/ReadVariableOp_4ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_4?
!gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell_12/strided_slice_4/stack?
#gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02%
#gru_cell_12/strided_slice_4/stack_1?
#gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_4/stack_2?
gru_cell_12/strided_slice_4StridedSlice$gru_cell_12/ReadVariableOp_4:value:0*gru_cell_12/strided_slice_4/stack:output:0,gru_cell_12/strided_slice_4/stack_1:output:0,gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell_12/strided_slice_4?
gru_cell_12/BiasAdd_1BiasAddgru_cell_12/MatMul_1:product:0$gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_1?
gru_cell_12/ReadVariableOp_5ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_5?
!gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02#
!gru_cell_12/strided_slice_5/stack?
#gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_12/strided_slice_5/stack_1?
#gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_5/stack_2?
gru_cell_12/strided_slice_5StridedSlice$gru_cell_12/ReadVariableOp_5:value:0*gru_cell_12/strided_slice_5/stack:output:0,gru_cell_12/strided_slice_5/stack_1:output:0,gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell_12/strided_slice_5?
gru_cell_12/BiasAdd_2BiasAddgru_cell_12/MatMul_2:product:0$gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_2?
gru_cell_12/ReadVariableOp_6ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_6?
!gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell_12/strided_slice_6/stack?
#gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#gru_cell_12/strided_slice_6/stack_1?
#gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_6/stack_2?
gru_cell_12/strided_slice_6StridedSlice$gru_cell_12/ReadVariableOp_6:value:0*gru_cell_12/strided_slice_6/stack:output:0,gru_cell_12/strided_slice_6/stack_1:output:0,gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_6?
gru_cell_12/MatMul_3MatMulzeros:output:0$gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_3?
gru_cell_12/ReadVariableOp_7ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_7?
!gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_7/stack?
#gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_7/stack_1?
#gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_7/stack_2?
gru_cell_12/strided_slice_7StridedSlice$gru_cell_12/ReadVariableOp_7:value:0*gru_cell_12/strided_slice_7/stack:output:0,gru_cell_12/strided_slice_7/stack_1:output:0,gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_7?
gru_cell_12/MatMul_4MatMulzeros:output:0$gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_4?
gru_cell_12/addAddV2gru_cell_12/BiasAdd:output:0gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/addk
gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Consto
gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_1?
gru_cell_12/MulMulgru_cell_12/add:z:0gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul?
gru_cell_12/Add_1AddV2gru_cell_12/Mul:z:0gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_1?
#gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#gru_cell_12/clip_by_value/Minimum/y?
!gru_cell_12/clip_by_value/MinimumMinimumgru_cell_12/Add_1:z:0,gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2#
!gru_cell_12/clip_by_value/Minimum
gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value/y?
gru_cell_12/clip_by_valueMaximum%gru_cell_12/clip_by_value/Minimum:z:0$gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value?
gru_cell_12/add_2AddV2gru_cell_12/BiasAdd_1:output:0gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_2o
gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Const_2o
gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_3?
gru_cell_12/Mul_1Mulgru_cell_12/add_2:z:0gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul_1?
gru_cell_12/Add_3AddV2gru_cell_12/Mul_1:z:0gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_3?
%gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gru_cell_12/clip_by_value_1/Minimum/y?
#gru_cell_12/clip_by_value_1/MinimumMinimumgru_cell_12/Add_3:z:0.gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2%
#gru_cell_12/clip_by_value_1/Minimum?
gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value_1/y?
gru_cell_12/clip_by_value_1Maximum'gru_cell_12/clip_by_value_1/Minimum:z:0&gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value_1?
gru_cell_12/mul_2Mulgru_cell_12/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_2?
gru_cell_12/ReadVariableOp_8ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_8?
!gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_8/stack?
#gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_8/stack_1?
#gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_8/stack_2?
gru_cell_12/strided_slice_8StridedSlice$gru_cell_12/ReadVariableOp_8:value:0*gru_cell_12/strided_slice_8/stack:output:0,gru_cell_12/strided_slice_8/stack_1:output:0,gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_8?
gru_cell_12/MatMul_5MatMulgru_cell_12/mul_2:z:0$gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_5?
gru_cell_12/add_4AddV2gru_cell_12/BiasAdd_2:output:0gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_4u
gru_cell_12/ReluRelugru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Relu?
gru_cell_12/mul_3Mulgru_cell_12/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_3k
gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_12/sub/x?
gru_cell_12/subSubgru_cell_12/sub/x:output:0gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/sub?
gru_cell_12/mul_4Mulgru_cell_12/sub:z:0gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_4?
gru_cell_12/add_5AddV2gru_cell_12/mul_3:z:0gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_5?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_12_readvariableop_resource%gru_cell_12_readvariableop_3_resource%gru_cell_12_readvariableop_6_resource*
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
bodyR
while_body_289935*
condR
while_cond_289934*8
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
NoOpNoOp^gru_cell_12/ReadVariableOp^gru_cell_12/ReadVariableOp_1^gru_cell_12/ReadVariableOp_2^gru_cell_12/ReadVariableOp_3^gru_cell_12/ReadVariableOp_4^gru_cell_12/ReadVariableOp_5^gru_cell_12/ReadVariableOp_6^gru_cell_12/ReadVariableOp_7^gru_cell_12/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
gru_cell_12/ReadVariableOpgru_cell_12/ReadVariableOp2<
gru_cell_12/ReadVariableOp_1gru_cell_12/ReadVariableOp_12<
gru_cell_12/ReadVariableOp_2gru_cell_12/ReadVariableOp_22<
gru_cell_12/ReadVariableOp_3gru_cell_12/ReadVariableOp_32<
gru_cell_12/ReadVariableOp_4gru_cell_12/ReadVariableOp_42<
gru_cell_12/ReadVariableOp_5gru_cell_12/ReadVariableOp_52<
gru_cell_12/ReadVariableOp_6gru_cell_12/ReadVariableOp_62<
gru_cell_12/ReadVariableOp_7gru_cell_12/ReadVariableOp_72<
gru_cell_12/ReadVariableOp_8gru_cell_12/ReadVariableOp_82
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
B__inference_gru_12_layer_call_and_return_conditional_losses_291894
inputs_06
#gru_cell_12_readvariableop_resource:	?H3
%gru_cell_12_readvariableop_3_resource:H7
%gru_cell_12_readvariableop_6_resource:H
identity??gru_cell_12/ReadVariableOp?gru_cell_12/ReadVariableOp_1?gru_cell_12/ReadVariableOp_2?gru_cell_12/ReadVariableOp_3?gru_cell_12/ReadVariableOp_4?gru_cell_12/ReadVariableOp_5?gru_cell_12/ReadVariableOp_6?gru_cell_12/ReadVariableOp_7?gru_cell_12/ReadVariableOp_8?whileF
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
gru_cell_12/ReadVariableOpReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp?
gru_cell_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
gru_cell_12/strided_slice/stack?
!gru_cell_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice/stack_1?
!gru_cell_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell_12/strided_slice/stack_2?
gru_cell_12/strided_sliceStridedSlice"gru_cell_12/ReadVariableOp:value:0(gru_cell_12/strided_slice/stack:output:0*gru_cell_12/strided_slice/stack_1:output:0*gru_cell_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice?
gru_cell_12/MatMulMatMulstrided_slice_2:output:0"gru_cell_12/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul?
gru_cell_12/ReadVariableOp_1ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_1?
!gru_cell_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_1/stack?
#gru_cell_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_1/stack_1?
#gru_cell_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_1/stack_2?
gru_cell_12/strided_slice_1StridedSlice$gru_cell_12/ReadVariableOp_1:value:0*gru_cell_12/strided_slice_1/stack:output:0,gru_cell_12/strided_slice_1/stack_1:output:0,gru_cell_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_1?
gru_cell_12/MatMul_1MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_1?
gru_cell_12/ReadVariableOp_2ReadVariableOp#gru_cell_12_readvariableop_resource*
_output_shapes
:	?H*
dtype02
gru_cell_12/ReadVariableOp_2?
!gru_cell_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_2/stack?
#gru_cell_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_2/stack_1?
#gru_cell_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_2/stack_2?
gru_cell_12/strided_slice_2StridedSlice$gru_cell_12/ReadVariableOp_2:value:0*gru_cell_12/strided_slice_2/stack:output:0,gru_cell_12/strided_slice_2/stack_1:output:0,gru_cell_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*

begin_mask*
end_mask2
gru_cell_12/strided_slice_2?
gru_cell_12/MatMul_2MatMulstrided_slice_2:output:0$gru_cell_12/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_2?
gru_cell_12/ReadVariableOp_3ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_3?
!gru_cell_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell_12/strided_slice_3/stack?
#gru_cell_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_1?
#gru_cell_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_3/stack_2?
gru_cell_12/strided_slice_3StridedSlice$gru_cell_12/ReadVariableOp_3:value:0*gru_cell_12/strided_slice_3/stack:output:0,gru_cell_12/strided_slice_3/stack_1:output:0,gru_cell_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru_cell_12/strided_slice_3?
gru_cell_12/BiasAddBiasAddgru_cell_12/MatMul:product:0$gru_cell_12/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd?
gru_cell_12/ReadVariableOp_4ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_4?
!gru_cell_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell_12/strided_slice_4/stack?
#gru_cell_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02%
#gru_cell_12/strided_slice_4/stack_1?
#gru_cell_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_4/stack_2?
gru_cell_12/strided_slice_4StridedSlice$gru_cell_12/ReadVariableOp_4:value:0*gru_cell_12/strided_slice_4/stack:output:0,gru_cell_12/strided_slice_4/stack_1:output:0,gru_cell_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru_cell_12/strided_slice_4?
gru_cell_12/BiasAdd_1BiasAddgru_cell_12/MatMul_1:product:0$gru_cell_12/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_1?
gru_cell_12/ReadVariableOp_5ReadVariableOp%gru_cell_12_readvariableop_3_resource*
_output_shapes
:H*
dtype02
gru_cell_12/ReadVariableOp_5?
!gru_cell_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:02#
!gru_cell_12/strided_slice_5/stack?
#gru_cell_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_12/strided_slice_5/stack_1?
#gru_cell_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_12/strided_slice_5/stack_2?
gru_cell_12/strided_slice_5StridedSlice$gru_cell_12/ReadVariableOp_5:value:0*gru_cell_12/strided_slice_5/stack:output:0,gru_cell_12/strided_slice_5/stack_1:output:0,gru_cell_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru_cell_12/strided_slice_5?
gru_cell_12/BiasAdd_2BiasAddgru_cell_12/MatMul_2:product:0$gru_cell_12/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/BiasAdd_2?
gru_cell_12/ReadVariableOp_6ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_6?
!gru_cell_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell_12/strided_slice_6/stack?
#gru_cell_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#gru_cell_12/strided_slice_6/stack_1?
#gru_cell_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_6/stack_2?
gru_cell_12/strided_slice_6StridedSlice$gru_cell_12/ReadVariableOp_6:value:0*gru_cell_12/strided_slice_6/stack:output:0,gru_cell_12/strided_slice_6/stack_1:output:0,gru_cell_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_6?
gru_cell_12/MatMul_3MatMulzeros:output:0$gru_cell_12/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_3?
gru_cell_12/ReadVariableOp_7ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_7?
!gru_cell_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!gru_cell_12/strided_slice_7/stack?
#gru_cell_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2%
#gru_cell_12/strided_slice_7/stack_1?
#gru_cell_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_7/stack_2?
gru_cell_12/strided_slice_7StridedSlice$gru_cell_12/ReadVariableOp_7:value:0*gru_cell_12/strided_slice_7/stack:output:0,gru_cell_12/strided_slice_7/stack_1:output:0,gru_cell_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_7?
gru_cell_12/MatMul_4MatMulzeros:output:0$gru_cell_12/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_4?
gru_cell_12/addAddV2gru_cell_12/BiasAdd:output:0gru_cell_12/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/addk
gru_cell_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Consto
gru_cell_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_1?
gru_cell_12/MulMulgru_cell_12/add:z:0gru_cell_12/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul?
gru_cell_12/Add_1AddV2gru_cell_12/Mul:z:0gru_cell_12/Const_1:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_1?
#gru_cell_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#gru_cell_12/clip_by_value/Minimum/y?
!gru_cell_12/clip_by_value/MinimumMinimumgru_cell_12/Add_1:z:0,gru_cell_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2#
!gru_cell_12/clip_by_value/Minimum
gru_cell_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value/y?
gru_cell_12/clip_by_valueMaximum%gru_cell_12/clip_by_value/Minimum:z:0$gru_cell_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value?
gru_cell_12/add_2AddV2gru_cell_12/BiasAdd_1:output:0gru_cell_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_2o
gru_cell_12/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru_cell_12/Const_2o
gru_cell_12/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru_cell_12/Const_3?
gru_cell_12/Mul_1Mulgru_cell_12/add_2:z:0gru_cell_12/Const_2:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Mul_1?
gru_cell_12/Add_3AddV2gru_cell_12/Mul_1:z:0gru_cell_12/Const_3:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Add_3?
%gru_cell_12/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gru_cell_12/clip_by_value_1/Minimum/y?
#gru_cell_12/clip_by_value_1/MinimumMinimumgru_cell_12/Add_3:z:0.gru_cell_12/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2%
#gru_cell_12/clip_by_value_1/Minimum?
gru_cell_12/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_cell_12/clip_by_value_1/y?
gru_cell_12/clip_by_value_1Maximum'gru_cell_12/clip_by_value_1/Minimum:z:0&gru_cell_12/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/clip_by_value_1?
gru_cell_12/mul_2Mulgru_cell_12/clip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_2?
gru_cell_12/ReadVariableOp_8ReadVariableOp%gru_cell_12_readvariableop_6_resource*
_output_shapes

:H*
dtype02
gru_cell_12/ReadVariableOp_8?
!gru_cell_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2#
!gru_cell_12/strided_slice_8/stack?
#gru_cell_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_12/strided_slice_8/stack_1?
#gru_cell_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_12/strided_slice_8/stack_2?
gru_cell_12/strided_slice_8StridedSlice$gru_cell_12/ReadVariableOp_8:value:0*gru_cell_12/strided_slice_8/stack:output:0,gru_cell_12/strided_slice_8/stack_1:output:0,gru_cell_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru_cell_12/strided_slice_8?
gru_cell_12/MatMul_5MatMulgru_cell_12/mul_2:z:0$gru_cell_12/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/MatMul_5?
gru_cell_12/add_4AddV2gru_cell_12/BiasAdd_2:output:0gru_cell_12/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_4u
gru_cell_12/ReluRelugru_cell_12/add_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/Relu?
gru_cell_12/mul_3Mulgru_cell_12/clip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_3k
gru_cell_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_12/sub/x?
gru_cell_12/subSubgru_cell_12/sub/x:output:0gru_cell_12/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/sub?
gru_cell_12/mul_4Mulgru_cell_12/sub:z:0gru_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/mul_4?
gru_cell_12/add_5AddV2gru_cell_12/mul_3:z:0gru_cell_12/mul_4:z:0*
T0*'
_output_shapes
:?????????2
gru_cell_12/add_5?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_12_readvariableop_resource%gru_cell_12_readvariableop_3_resource%gru_cell_12_readvariableop_6_resource*
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
bodyR
while_body_291756*
condR
while_cond_291755*8
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
NoOpNoOp^gru_cell_12/ReadVariableOp^gru_cell_12/ReadVariableOp_1^gru_cell_12/ReadVariableOp_2^gru_cell_12/ReadVariableOp_3^gru_cell_12/ReadVariableOp_4^gru_cell_12/ReadVariableOp_5^gru_cell_12/ReadVariableOp_6^gru_cell_12/ReadVariableOp_7^gru_cell_12/ReadVariableOp_8^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 28
gru_cell_12/ReadVariableOpgru_cell_12/ReadVariableOp2<
gru_cell_12/ReadVariableOp_1gru_cell_12/ReadVariableOp_12<
gru_cell_12/ReadVariableOp_2gru_cell_12/ReadVariableOp_22<
gru_cell_12/ReadVariableOp_3gru_cell_12/ReadVariableOp_32<
gru_cell_12/ReadVariableOp_4gru_cell_12/ReadVariableOp_42<
gru_cell_12/ReadVariableOp_5gru_cell_12/ReadVariableOp_52<
gru_cell_12/ReadVariableOp_6gru_cell_12/ReadVariableOp_62<
gru_cell_12/ReadVariableOp_7gru_cell_12/ReadVariableOp_72<
gru_cell_12/ReadVariableOp_8gru_cell_12/ReadVariableOp_82
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?

?
D__inference_dense_25_layer_call_and_return_conditional_losses_293145

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
?"
?
while_body_288922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_12_288944_0:	?H(
while_gru_cell_12_288946_0:H,
while_gru_cell_12_288948_0:H
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_12_288944:	?H&
while_gru_cell_12_288946:H*
while_gru_cell_12_288948:H??)while/gru_cell_12/StatefulPartitionedCall?
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
)while/gru_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_12_288944_0while_gru_cell_12_288946_0while_gru_cell_12_288948_0*
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
GPU2*0J 8? *P
fKRI
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_2889092+
)while/gru_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_12/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"6
while_gru_cell_12_288944while_gru_cell_12_288944_0"6
while_gru_cell_12_288946while_gru_cell_12_288946_0"6
while_gru_cell_12_288948while_gru_cell_12_288948_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2V
)while/gru_cell_12/StatefulPartitionedCall)while/gru_cell_12/StatefulPartitionedCall: 
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
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
conv1d_24_input<
!serving_default_conv1d_24_input:0?????????K
time_distributed_254
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(cell
)
state_spec
*trainable_variables
+regularization_losses
,	variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	2layer
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	7layer
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<iter

=beta_1

>beta_2
	?decay
@learning_rate
Amomentum_cachem?m?m?m?Bm?Cm?Dm?Em?Fm?Gm?Hm?v?v?v?v?Bv?Cv?Dv?Ev?Fv?Gv?Hv?"
	optimizer
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
?

Ilayers
trainable_variables
regularization_losses
Jlayer_regularization_losses
Kmetrics
	variables
Lnon_trainable_variables
Mlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$ 2conv1d_24/kernel
: 2conv1d_24/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

Nlayers
trainable_variables
regularization_losses
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
Rnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$ @2conv1d_25/kernel
:@2conv1d_25/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

Slayers
trainable_variables
regularization_losses
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
Wnon_trainable_variables
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
	variables

Xlayers
trainable_variables
regularization_losses
Ylayer_regularization_losses
Zmetrics
[layer_metrics
\non_trainable_variables
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
 	variables

]layers
!trainable_variables
"regularization_losses
^layer_regularization_losses
_metrics
`layer_metrics
anon_trainable_variables
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
$	variables

blayers
%trainable_variables
&regularization_losses
clayer_regularization_losses
dmetrics
elayer_metrics
fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Bkernel
Crecurrent_kernel
Dbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
B0
C1
D2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
B0
C1
D2"
trackable_list_wrapper
?
*trainable_variables

klayers

lstates
+regularization_losses
mlayer_regularization_losses
nmetrics
,	variables
onon_trainable_variables
player_metrics
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
.	variables

qlayers
/trainable_variables
0regularization_losses
rlayer_regularization_losses
smetrics
tlayer_metrics
unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Ekernel
Fbias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3	variables

zlayers
4trainable_variables
5regularization_losses
{layer_regularization_losses
|metrics
}layer_metrics
~non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Gkernel
Hbias
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8	variables
?layers
9trainable_variables
:regularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
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
,:*	?H2gru_12/gru_cell_12/kernel
5:3H2#gru_12/gru_cell_12/recurrent_kernel
%:#H2gru_12/gru_cell_12/bias
,:* 2time_distributed_24/kernel
&:$ 2time_distributed_24/bias
,:* 2time_distributed_25/kernel
&:$2time_distributed_25/bias
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
trackable_list_wrapper
0
?0
?1"
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
 "
trackable_list_wrapper
?
g	variables
?layers
htrainable_variables
iregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
(0"
trackable_list_wrapper
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
trackable_dict_wrapper
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
 "
trackable_list_wrapper
?
v	variables
?layers
wtrainable_variables
xregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
?
	variables
?layers
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
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
,:* 2Nadam/conv1d_24/kernel/m
":  2Nadam/conv1d_24/bias/m
,:* @2Nadam/conv1d_25/kernel/m
": @2Nadam/conv1d_25/bias/m
2:0	?H2!Nadam/gru_12/gru_cell_12/kernel/m
;:9H2+Nadam/gru_12/gru_cell_12/recurrent_kernel/m
+:)H2Nadam/gru_12/gru_cell_12/bias/m
2:0 2"Nadam/time_distributed_24/kernel/m
,:* 2 Nadam/time_distributed_24/bias/m
2:0 2"Nadam/time_distributed_25/kernel/m
,:*2 Nadam/time_distributed_25/bias/m
,:* 2Nadam/conv1d_24/kernel/v
":  2Nadam/conv1d_24/bias/v
,:* @2Nadam/conv1d_25/kernel/v
": @2Nadam/conv1d_25/bias/v
2:0	?H2!Nadam/gru_12/gru_cell_12/kernel/v
;:9H2+Nadam/gru_12/gru_cell_12/recurrent_kernel/v
+:)H2Nadam/gru_12/gru_cell_12/bias/v
2:0 2"Nadam/time_distributed_24/kernel/v
,:* 2 Nadam/time_distributed_24/bias/v
2:0 2"Nadam/time_distributed_25/kernel/v
,:*2 Nadam/time_distributed_25/bias/v
?2?
.__inference_sequential_12_layer_call_fn_290156
.__inference_sequential_12_layer_call_fn_290817
.__inference_sequential_12_layer_call_fn_290844
.__inference_sequential_12_layer_call_fn_290677?
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_291159
I__inference_sequential_12_layer_call_and_return_conditional_losses_291481
I__inference_sequential_12_layer_call_and_return_conditional_losses_290716
I__inference_sequential_12_layer_call_and_return_conditional_losses_290755?
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
!__inference__wrapped_model_288729conv1d_24_input"?
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
?2?
*__inference_conv1d_24_layer_call_fn_291490?
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
E__inference_conv1d_24_layer_call_and_return_conditional_losses_291506?
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
*__inference_conv1d_25_layer_call_fn_291515?
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
E__inference_conv1d_25_layer_call_and_return_conditional_losses_291531?
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
1__inference_max_pooling1d_12_layer_call_fn_291536
1__inference_max_pooling1d_12_layer_call_fn_291541?
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
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_291549
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_291557?
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
+__inference_flatten_12_layer_call_fn_291562?
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
F__inference_flatten_12_layer_call_and_return_conditional_losses_291568?
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
1__inference_repeat_vector_12_layer_call_fn_291573
1__inference_repeat_vector_12_layer_call_fn_291578?
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
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_291586
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_291594?
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
'__inference_gru_12_layer_call_fn_291605
'__inference_gru_12_layer_call_fn_291616
'__inference_gru_12_layer_call_fn_291627
'__inference_gru_12_layer_call_fn_291638?
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
?2?
B__inference_gru_12_layer_call_and_return_conditional_losses_291894
B__inference_gru_12_layer_call_and_return_conditional_losses_292150
B__inference_gru_12_layer_call_and_return_conditional_losses_292406
B__inference_gru_12_layer_call_and_return_conditional_losses_292662?
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
+__inference_dropout_12_layer_call_fn_292667
+__inference_dropout_12_layer_call_fn_292672?
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
F__inference_dropout_12_layer_call_and_return_conditional_losses_292677
F__inference_dropout_12_layer_call_and_return_conditional_losses_292689?
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
4__inference_time_distributed_24_layer_call_fn_292698
4__inference_time_distributed_24_layer_call_fn_292707
4__inference_time_distributed_24_layer_call_fn_292716
4__inference_time_distributed_24_layer_call_fn_292725?
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
?2?
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292746
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292767
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292781
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292795?
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
4__inference_time_distributed_25_layer_call_fn_292804
4__inference_time_distributed_25_layer_call_fn_292813
4__inference_time_distributed_25_layer_call_fn_292822
4__inference_time_distributed_25_layer_call_fn_292831?
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
?2?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292852
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292873
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292887
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292901?
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
$__inference_signature_wrapper_290790conv1d_24_input"?
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
,__inference_gru_cell_12_layer_call_fn_292915
,__inference_gru_cell_12_layer_call_fn_292929?
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
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_293018
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_293107?
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
)__inference_dense_24_layer_call_fn_293116?
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
D__inference_dense_24_layer_call_and_return_conditional_losses_293126?
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
)__inference_dense_25_layer_call_fn_293135?
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
D__inference_dense_25_layer_call_and_return_conditional_losses_293145?
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
!__inference__wrapped_model_288729?BDCEFGH<?9
2?/
-?*
conv1d_24_input?????????
? "M?J
H
time_distributed_251?.
time_distributed_25??????????
E__inference_conv1d_24_layer_call_and_return_conditional_losses_291506d3?0
)?&
$?!
inputs?????????
? ")?&
?
0????????? 
? ?
*__inference_conv1d_24_layer_call_fn_291490W3?0
)?&
$?!
inputs?????????
? "?????????? ?
E__inference_conv1d_25_layer_call_and_return_conditional_losses_291531d3?0
)?&
$?!
inputs????????? 
? ")?&
?
0?????????@
? ?
*__inference_conv1d_25_layer_call_fn_291515W3?0
)?&
$?!
inputs????????? 
? "??????????@?
D__inference_dense_24_layer_call_and_return_conditional_losses_293126\EF/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? |
)__inference_dense_24_layer_call_fn_293116OEF/?,
%?"
 ?
inputs?????????
? "?????????? ?
D__inference_dense_25_layer_call_and_return_conditional_losses_293145\GH/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense_25_layer_call_fn_293135OGH/?,
%?"
 ?
inputs????????? 
? "???????????
F__inference_dropout_12_layer_call_and_return_conditional_losses_292677d7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
F__inference_dropout_12_layer_call_and_return_conditional_losses_292689d7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
+__inference_dropout_12_layer_call_fn_292667W7?4
-?*
$?!
inputs?????????
p 
? "???????????
+__inference_dropout_12_layer_call_fn_292672W7?4
-?*
$?!
inputs?????????
p
? "???????????
F__inference_flatten_12_layer_call_and_return_conditional_losses_291568]3?0
)?&
$?!
inputs?????????@
? "&?#
?
0??????????
? 
+__inference_flatten_12_layer_call_fn_291562P3?0
)?&
$?!
inputs?????????@
? "????????????
B__inference_gru_12_layer_call_and_return_conditional_losses_291894?BDCP?M
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
B__inference_gru_12_layer_call_and_return_conditional_losses_292150?BDCP?M
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
B__inference_gru_12_layer_call_and_return_conditional_losses_292406rBDC@?=
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
B__inference_gru_12_layer_call_and_return_conditional_losses_292662rBDC@?=
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
'__inference_gru_12_layer_call_fn_291605~BDCP?M
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
'__inference_gru_12_layer_call_fn_291616~BDCP?M
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
'__inference_gru_12_layer_call_fn_291627eBDC@?=
6?3
%?"
inputs??????????

 
p 

 
? "???????????
'__inference_gru_12_layer_call_fn_291638eBDC@?=
6?3
%?"
inputs??????????

 
p

 
? "???????????
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_293018?BDC]?Z
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
G__inference_gru_cell_12_layer_call_and_return_conditional_losses_293107?BDC]?Z
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
,__inference_gru_cell_12_layer_call_fn_292915?BDC]?Z
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
,__inference_gru_cell_12_layer_call_fn_292929?BDC]?Z
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
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_291549?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
L__inference_max_pooling1d_12_layer_call_and_return_conditional_losses_291557`3?0
)?&
$?!
inputs?????????@
? ")?&
?
0?????????@
? ?
1__inference_max_pooling1d_12_layer_call_fn_291536wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
1__inference_max_pooling1d_12_layer_call_fn_291541S3?0
)?&
$?!
inputs?????????@
? "??????????@?
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_291586n8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
L__inference_repeat_vector_12_layer_call_and_return_conditional_losses_291594^0?-
&?#
!?
inputs??????????
? "*?'
 ?
0??????????
? ?
1__inference_repeat_vector_12_layer_call_fn_291573a8?5
.?+
)?&
inputs??????????????????
? "%?"???????????????????
1__inference_repeat_vector_12_layer_call_fn_291578Q0?-
&?#
!?
inputs??????????
? "????????????
I__inference_sequential_12_layer_call_and_return_conditional_losses_290716~BDCEFGHD?A
:?7
-?*
conv1d_24_input?????????
p 

 
? ")?&
?
0?????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_290755~BDCEFGHD?A
:?7
-?*
conv1d_24_input?????????
p

 
? ")?&
?
0?????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_291159uBDCEFGH;?8
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_291481uBDCEFGH;?8
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
.__inference_sequential_12_layer_call_fn_290156qBDCEFGHD?A
:?7
-?*
conv1d_24_input?????????
p 

 
? "???????????
.__inference_sequential_12_layer_call_fn_290677qBDCEFGHD?A
:?7
-?*
conv1d_24_input?????????
p

 
? "???????????
.__inference_sequential_12_layer_call_fn_290817hBDCEFGH;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
.__inference_sequential_12_layer_call_fn_290844hBDCEFGH;?8
1?.
$?!
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_290790?BDCEFGHO?L
? 
E?B
@
conv1d_24_input-?*
conv1d_24_input?????????"M?J
H
time_distributed_251?.
time_distributed_25??????????
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292746~EFD?A
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
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292767~EFD?A
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
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292781lEF;?8
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
O__inference_time_distributed_24_layer_call_and_return_conditional_losses_292795lEF;?8
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
4__inference_time_distributed_24_layer_call_fn_292698qEFD?A
:?7
-?*
inputs??????????????????
p 

 
? "%?"?????????????????? ?
4__inference_time_distributed_24_layer_call_fn_292707qEFD?A
:?7
-?*
inputs??????????????????
p

 
? "%?"?????????????????? ?
4__inference_time_distributed_24_layer_call_fn_292716_EF;?8
1?.
$?!
inputs?????????
p 

 
? "?????????? ?
4__inference_time_distributed_24_layer_call_fn_292725_EF;?8
1?.
$?!
inputs?????????
p

 
? "?????????? ?
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292852~GHD?A
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
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292873~GHD?A
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
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292887lGH;?8
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
O__inference_time_distributed_25_layer_call_and_return_conditional_losses_292901lGH;?8
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
4__inference_time_distributed_25_layer_call_fn_292804qGHD?A
:?7
-?*
inputs?????????????????? 
p 

 
? "%?"???????????????????
4__inference_time_distributed_25_layer_call_fn_292813qGHD?A
:?7
-?*
inputs?????????????????? 
p

 
? "%?"???????????????????
4__inference_time_distributed_25_layer_call_fn_292822_GH;?8
1?.
$?!
inputs????????? 
p 

 
? "???????????
4__inference_time_distributed_25_layer_call_fn_292831_GH;?8
1?.
$?!
inputs????????? 
p

 
? "??????????