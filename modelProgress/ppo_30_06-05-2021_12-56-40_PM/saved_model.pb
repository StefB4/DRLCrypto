ì	
¿£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8Ø

Policy_mu_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namePolicy_mu_readout/kernel

,Policy_mu_readout/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_readout/kernel*
_output_shapes

: *
dtype0

Policy_mu_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namePolicy_mu_readout/bias
}
*Policy_mu_readout/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_readout/bias*
_output_shapes
:*
dtype0

Policy_sigma_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_namePolicy_sigma_readout/kernel

/Policy_sigma_readout/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_readout/kernel*
_output_shapes

: *
dtype0

Policy_sigma_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namePolicy_sigma_readout/bias

-Policy_sigma_readout/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_readout/bias*
_output_shapes
:*
dtype0

Value_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameValue_readout/kernel
}
(Value_readout/kernel/Read/ReadVariableOpReadVariableOpValue_readout/kernel*
_output_shapes

: *
dtype0
|
Value_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameValue_readout/bias
u
&Value_readout/bias/Read/ReadVariableOpReadVariableOpValue_readout/bias*
_output_shapes
:*
dtype0

Policy_mu_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_namePolicy_mu_0/kernel
y
&Policy_mu_0/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_0/kernel*
_output_shapes

: *
dtype0
x
Policy_mu_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namePolicy_mu_0/bias
q
$Policy_mu_0/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_0/bias*
_output_shapes
: *
dtype0

Policy_mu_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *#
shared_namePolicy_mu_1/kernel
y
&Policy_mu_1/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_1/kernel*
_output_shapes

:  *
dtype0
x
Policy_mu_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namePolicy_mu_1/bias
q
$Policy_mu_1/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_1/bias*
_output_shapes
: *
dtype0

Policy_mu_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *#
shared_namePolicy_mu_2/kernel
y
&Policy_mu_2/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_2/kernel*
_output_shapes

:  *
dtype0
x
Policy_mu_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namePolicy_mu_2/bias
q
$Policy_mu_2/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_2/bias*
_output_shapes
: *
dtype0

Policy_sigma_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_namePolicy_sigma_0/kernel

)Policy_sigma_0/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_0/kernel*
_output_shapes

: *
dtype0
~
Policy_sigma_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namePolicy_sigma_0/bias
w
'Policy_sigma_0/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_0/bias*
_output_shapes
: *
dtype0

Policy_sigma_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_namePolicy_sigma_1/kernel

)Policy_sigma_1/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_1/kernel*
_output_shapes

:  *
dtype0
~
Policy_sigma_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namePolicy_sigma_1/bias
w
'Policy_sigma_1/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_1/bias*
_output_shapes
: *
dtype0

Policy_sigma_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_namePolicy_sigma_2/kernel

)Policy_sigma_2/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_2/kernel*
_output_shapes

:  *
dtype0
~
Policy_sigma_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namePolicy_sigma_2/bias
w
'Policy_sigma_2/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_2/bias*
_output_shapes
: *
dtype0

Value_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameValue_layer_0/kernel
}
(Value_layer_0/kernel/Read/ReadVariableOpReadVariableOpValue_layer_0/kernel*
_output_shapes

: *
dtype0
|
Value_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameValue_layer_0/bias
u
&Value_layer_0/bias/Read/ReadVariableOpReadVariableOpValue_layer_0/bias*
_output_shapes
: *
dtype0

Value_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *%
shared_nameValue_layer_1/kernel
}
(Value_layer_1/kernel/Read/ReadVariableOpReadVariableOpValue_layer_1/kernel*
_output_shapes

:  *
dtype0
|
Value_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameValue_layer_1/bias
u
&Value_layer_1/bias/Read/ReadVariableOpReadVariableOpValue_layer_1/bias*
_output_shapes
: *
dtype0

Value_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *%
shared_nameValue_layer_2/kernel
}
(Value_layer_2/kernel/Read/ReadVariableOpReadVariableOpValue_layer_2/kernel*
_output_shapes

:  *
dtype0
|
Value_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameValue_layer_2/bias
u
&Value_layer_2/bias/Read/ReadVariableOpReadVariableOpValue_layer_2/bias*
_output_shapes
: *
dtype0

NoOpNoOp
å7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 7
value7B7 B7
È
mu_layer

readout_mu
sigma_layer
readout_sigma
value_layer
readout_value
trainable_variables
	variables
	regularization_losses

	keras_api

signatures

0
1
2
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1
2
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1
 2
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
¶
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23
¶
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23
 
­
9metrics
:non_trainable_variables
;layer_regularization_losses
<layer_metrics
trainable_variables
	variables
	regularization_losses

=layers
 
h

'kernel
(bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

)kernel
*bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
h

+kernel
,bias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
ZX
VARIABLE_VALUEPolicy_mu_readout/kernel,readout_mu/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_readout/bias*readout_mu/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Jmetrics
Knon_trainable_variables
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
	variables
regularization_losses

Nlayers
h

-kernel
.bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
h

/kernel
0bias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
h

1kernel
2bias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
`^
VARIABLE_VALUEPolicy_sigma_readout/kernel/readout_sigma/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEPolicy_sigma_readout/bias-readout_sigma/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
[metrics
\non_trainable_variables
]layer_metrics
^layer_regularization_losses
trainable_variables
	variables
regularization_losses

_layers
h

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
h

5kernel
6bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
h

7kernel
8bias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
YW
VARIABLE_VALUEValue_readout/kernel/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEValue_readout/bias-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
­
lmetrics
mnon_trainable_variables
nlayer_metrics
olayer_regularization_losses
#trainable_variables
$	variables
%regularization_losses

players
XV
VARIABLE_VALUEPolicy_mu_0/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_0/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEPolicy_mu_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEPolicy_mu_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEPolicy_sigma_0/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEPolicy_sigma_0/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEPolicy_sigma_1/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEPolicy_sigma_1/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEPolicy_sigma_2/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEPolicy_sigma_2/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEValue_layer_0/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEValue_layer_0/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEValue_layer_1/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEValue_layer_1/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEValue_layer_2/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEValue_layer_2/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
V
0
1
2
3
4
5
6
7
8
9
 10
11

'0
(1

'0
(1
 
­
qmetrics
rnon_trainable_variables
slayer_metrics
tlayer_regularization_losses
>trainable_variables
?	variables
@regularization_losses

ulayers

)0
*1

)0
*1
 
­
vmetrics
wnon_trainable_variables
xlayer_metrics
ylayer_regularization_losses
Btrainable_variables
C	variables
Dregularization_losses

zlayers

+0
,1

+0
,1
 
­
{metrics
|non_trainable_variables
}layer_metrics
~layer_regularization_losses
Ftrainable_variables
G	variables
Hregularization_losses

layers
 
 
 
 
 

-0
.1

-0
.1
 
²
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
Otrainable_variables
P	variables
Qregularization_losses
layers

/0
01

/0
01
 
²
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
Strainable_variables
T	variables
Uregularization_losses
layers

10
21

10
21
 
²
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
Wtrainable_variables
X	variables
Yregularization_losses
layers
 
 
 
 
 

30
41

30
41
 
²
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
`trainable_variables
a	variables
bregularization_losses
layers

50
61

50
61
 
²
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
dtrainable_variables
e	variables
fregularization_losses
layers

70
81

70
81
 
²
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
htrainable_variables
i	variables
jregularization_losses
layers
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
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ö
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Policy_mu_0/kernelPolicy_mu_0/biasPolicy_mu_1/kernelPolicy_mu_1/biasPolicy_mu_2/kernelPolicy_mu_2/biasPolicy_sigma_0/kernelPolicy_sigma_0/biasPolicy_sigma_1/kernelPolicy_sigma_1/biasPolicy_sigma_2/kernelPolicy_sigma_2/biasValue_layer_0/kernelValue_layer_0/biasValue_layer_1/kernelValue_layer_1/biasValue_layer_2/kernelValue_layer_2/biasPolicy_mu_readout/kernelPolicy_mu_readout/biasPolicy_sigma_readout/kernelPolicy_sigma_readout/biasValue_readout/kernelValue_readout/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:::ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_11599660
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,Policy_mu_readout/kernel/Read/ReadVariableOp*Policy_mu_readout/bias/Read/ReadVariableOp/Policy_sigma_readout/kernel/Read/ReadVariableOp-Policy_sigma_readout/bias/Read/ReadVariableOp(Value_readout/kernel/Read/ReadVariableOp&Value_readout/bias/Read/ReadVariableOp&Policy_mu_0/kernel/Read/ReadVariableOp$Policy_mu_0/bias/Read/ReadVariableOp&Policy_mu_1/kernel/Read/ReadVariableOp$Policy_mu_1/bias/Read/ReadVariableOp&Policy_mu_2/kernel/Read/ReadVariableOp$Policy_mu_2/bias/Read/ReadVariableOp)Policy_sigma_0/kernel/Read/ReadVariableOp'Policy_sigma_0/bias/Read/ReadVariableOp)Policy_sigma_1/kernel/Read/ReadVariableOp'Policy_sigma_1/bias/Read/ReadVariableOp)Policy_sigma_2/kernel/Read/ReadVariableOp'Policy_sigma_2/bias/Read/ReadVariableOp(Value_layer_0/kernel/Read/ReadVariableOp&Value_layer_0/bias/Read/ReadVariableOp(Value_layer_1/kernel/Read/ReadVariableOp&Value_layer_1/bias/Read/ReadVariableOp(Value_layer_2/kernel/Read/ReadVariableOp&Value_layer_2/bias/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_11600174
¸
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamePolicy_mu_readout/kernelPolicy_mu_readout/biasPolicy_sigma_readout/kernelPolicy_sigma_readout/biasValue_readout/kernelValue_readout/biasPolicy_mu_0/kernelPolicy_mu_0/biasPolicy_mu_1/kernelPolicy_mu_1/biasPolicy_mu_2/kernelPolicy_mu_2/biasPolicy_sigma_0/kernelPolicy_sigma_0/biasPolicy_sigma_1/kernelPolicy_sigma_1/biasPolicy_sigma_2/kernelPolicy_sigma_2/biasValue_layer_0/kernelValue_layer_0/biasValue_layer_1/kernelValue_layer_1/biasValue_layer_2/kernelValue_layer_2/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_11600256ðÖ
®
±
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_11599928

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í

1__inference_Policy_sigma_2_layer_call_fn_11600017

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_115993622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_Value_layer_1_layer_call_fn_11600057

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_115994162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç

.__inference_Policy_mu_0_layer_call_fn_11599917

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_115992272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
±
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_11599254

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¶
ö
)__inference_a2c_33_layer_call_fn_11599601
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:::ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_a2c_33_layer_call_and_return_conditional_losses_115995432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ø
·
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_11599850

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_Value_layer_2_layer_call_fn_11600077

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_115994432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
±
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_11599948

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±
´
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_11599968

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
³
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_11599389

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

.__inference_Policy_mu_1_layer_call_fn_11599937

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_115992542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ù

7__inference_Policy_sigma_readout_layer_call_fn_11599878

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_115994962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÿP


D__inference_a2c_33_layer_call_and_return_conditional_losses_11599543
input_1
policy_mu_0_11599238
policy_mu_0_11599240
policy_mu_1_11599265
policy_mu_1_11599267
policy_mu_2_11599292
policy_mu_2_11599294
policy_sigma_0_11599319
policy_sigma_0_11599321
policy_sigma_1_11599346
policy_sigma_1_11599348
policy_sigma_2_11599373
policy_sigma_2_11599375
value_layer_0_11599400
value_layer_0_11599402
value_layer_1_11599427
value_layer_1_11599429
value_layer_2_11599454
value_layer_2_11599456
policy_mu_readout_11599480
policy_mu_readout_11599482!
policy_sigma_readout_11599507!
policy_sigma_readout_11599509
value_readout_11599535
value_readout_11599537
identity

identity_1

identity_2¢#Policy_mu_0/StatefulPartitionedCall¢#Policy_mu_1/StatefulPartitionedCall¢#Policy_mu_2/StatefulPartitionedCall¢)Policy_mu_readout/StatefulPartitionedCall¢&Policy_sigma_0/StatefulPartitionedCall¢&Policy_sigma_1/StatefulPartitionedCall¢&Policy_sigma_2/StatefulPartitionedCall¢,Policy_sigma_readout/StatefulPartitionedCall¢%Value_layer_0/StatefulPartitionedCall¢%Value_layer_1/StatefulPartitionedCall¢%Value_layer_2/StatefulPartitionedCall¢%Value_readout/StatefulPartitionedCallª
#Policy_mu_0/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_mu_0_11599238policy_mu_0_11599240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_115992272%
#Policy_mu_0/StatefulPartitionedCallÏ
#Policy_mu_1/StatefulPartitionedCallStatefulPartitionedCall,Policy_mu_0/StatefulPartitionedCall:output:0policy_mu_1_11599265policy_mu_1_11599267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_115992542%
#Policy_mu_1/StatefulPartitionedCallÏ
#Policy_mu_2/StatefulPartitionedCallStatefulPartitionedCall,Policy_mu_1/StatefulPartitionedCall:output:0policy_mu_2_11599292policy_mu_2_11599294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_115992812%
#Policy_mu_2/StatefulPartitionedCall¹
&Policy_sigma_0/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_sigma_0_11599319policy_sigma_0_11599321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_115993082(
&Policy_sigma_0/StatefulPartitionedCallá
&Policy_sigma_1/StatefulPartitionedCallStatefulPartitionedCall/Policy_sigma_0/StatefulPartitionedCall:output:0policy_sigma_1_11599346policy_sigma_1_11599348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_115993352(
&Policy_sigma_1/StatefulPartitionedCallá
&Policy_sigma_2/StatefulPartitionedCallStatefulPartitionedCall/Policy_sigma_1/StatefulPartitionedCall:output:0policy_sigma_2_11599373policy_sigma_2_11599375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_115993622(
&Policy_sigma_2/StatefulPartitionedCall´
%Value_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_1value_layer_0_11599400value_layer_0_11599402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_115993892'
%Value_layer_0/StatefulPartitionedCallÛ
%Value_layer_1/StatefulPartitionedCallStatefulPartitionedCall.Value_layer_0/StatefulPartitionedCall:output:0value_layer_1_11599427value_layer_1_11599429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_115994162'
%Value_layer_1/StatefulPartitionedCallÛ
%Value_layer_2/StatefulPartitionedCallStatefulPartitionedCall.Value_layer_1/StatefulPartitionedCall:output:0value_layer_2_11599454value_layer_2_11599456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_115994432'
%Value_layer_2/StatefulPartitionedCallí
)Policy_mu_readout/StatefulPartitionedCallStatefulPartitionedCall,Policy_mu_2/StatefulPartitionedCall:output:0policy_mu_readout_11599480policy_mu_readout_11599482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_115994692+
)Policy_mu_readout/StatefulPartitionedCallt
SqueezeSqueeze2Policy_mu_readout/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2	
Squeezeÿ
,Policy_sigma_readout/StatefulPartitionedCallStatefulPartitionedCall/Policy_sigma_2/StatefulPartitionedCall:output:0policy_sigma_readout_11599507policy_sigma_readout_11599509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_115994962.
,Policy_sigma_readout/StatefulPartitionedCallz
AbsAbs5Policy_sigma_readout/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
AbsM
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1Û
%Value_readout/StatefulPartitionedCallStatefulPartitionedCall.Value_layer_2/StatefulPartitionedCall:output:0value_readout_11599535value_readout_11599537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_readout_layer_call_and_return_conditional_losses_115995242'
%Value_readout/StatefulPartitionedCall½
IdentityIdentitySqueeze:output:0$^Policy_mu_0/StatefulPartitionedCall$^Policy_mu_1/StatefulPartitionedCall$^Policy_mu_2/StatefulPartitionedCall*^Policy_mu_readout/StatefulPartitionedCall'^Policy_sigma_0/StatefulPartitionedCall'^Policy_sigma_1/StatefulPartitionedCall'^Policy_sigma_2/StatefulPartitionedCall-^Policy_sigma_readout/StatefulPartitionedCall&^Value_layer_0/StatefulPartitionedCall&^Value_layer_1/StatefulPartitionedCall&^Value_layer_2/StatefulPartitionedCall&^Value_readout/StatefulPartitionedCall*
T0*
_output_shapes
:2

IdentityÃ

Identity_1IdentitySqueeze_1:output:0$^Policy_mu_0/StatefulPartitionedCall$^Policy_mu_1/StatefulPartitionedCall$^Policy_mu_2/StatefulPartitionedCall*^Policy_mu_readout/StatefulPartitionedCall'^Policy_sigma_0/StatefulPartitionedCall'^Policy_sigma_1/StatefulPartitionedCall'^Policy_sigma_2/StatefulPartitionedCall-^Policy_sigma_readout/StatefulPartitionedCall&^Value_layer_0/StatefulPartitionedCall&^Value_layer_1/StatefulPartitionedCall&^Value_layer_2/StatefulPartitionedCall&^Value_readout/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1î

Identity_2Identity.Value_readout/StatefulPartitionedCall:output:0$^Policy_mu_0/StatefulPartitionedCall$^Policy_mu_1/StatefulPartitionedCall$^Policy_mu_2/StatefulPartitionedCall*^Policy_mu_readout/StatefulPartitionedCall'^Policy_sigma_0/StatefulPartitionedCall'^Policy_sigma_1/StatefulPartitionedCall'^Policy_sigma_2/StatefulPartitionedCall-^Policy_sigma_readout/StatefulPartitionedCall&^Value_layer_0/StatefulPartitionedCall&^Value_layer_1/StatefulPartitionedCall&^Value_layer_2/StatefulPartitionedCall&^Value_readout/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::2J
#Policy_mu_0/StatefulPartitionedCall#Policy_mu_0/StatefulPartitionedCall2J
#Policy_mu_1/StatefulPartitionedCall#Policy_mu_1/StatefulPartitionedCall2J
#Policy_mu_2/StatefulPartitionedCall#Policy_mu_2/StatefulPartitionedCall2V
)Policy_mu_readout/StatefulPartitionedCall)Policy_mu_readout/StatefulPartitionedCall2P
&Policy_sigma_0/StatefulPartitionedCall&Policy_sigma_0/StatefulPartitionedCall2P
&Policy_sigma_1/StatefulPartitionedCall&Policy_sigma_1/StatefulPartitionedCall2P
&Policy_sigma_2/StatefulPartitionedCall&Policy_sigma_2/StatefulPartitionedCall2\
,Policy_sigma_readout/StatefulPartitionedCall,Policy_sigma_readout/StatefulPartitionedCall2N
%Value_layer_0/StatefulPartitionedCall%Value_layer_0/StatefulPartitionedCall2N
%Value_layer_1/StatefulPartitionedCall%Value_layer_1/StatefulPartitionedCall2N
%Value_layer_2/StatefulPartitionedCall%Value_layer_2/StatefulPartitionedCall2N
%Value_readout/StatefulPartitionedCall%Value_readout/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
í

1__inference_Policy_sigma_0_layer_call_fn_11599977

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_115993082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
³
K__inference_Value_readout_layer_call_and_return_conditional_losses_11599524

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
ü
#__inference__wrapped_model_11599212
input_1
a2c_33_11599158
a2c_33_11599160
a2c_33_11599162
a2c_33_11599164
a2c_33_11599166
a2c_33_11599168
a2c_33_11599170
a2c_33_11599172
a2c_33_11599174
a2c_33_11599176
a2c_33_11599178
a2c_33_11599180
a2c_33_11599182
a2c_33_11599184
a2c_33_11599186
a2c_33_11599188
a2c_33_11599190
a2c_33_11599192
a2c_33_11599194
a2c_33_11599196
a2c_33_11599198
a2c_33_11599200
a2c_33_11599202
a2c_33_11599204
identity

identity_1

identity_2¢a2c_33/StatefulPartitionedCall
a2c_33/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_33_11599158a2c_33_11599160a2c_33_11599162a2c_33_11599164a2c_33_11599166a2c_33_11599168a2c_33_11599170a2c_33_11599172a2c_33_11599174a2c_33_11599176a2c_33_11599178a2c_33_11599180a2c_33_11599182a2c_33_11599184a2c_33_11599186a2c_33_11599188a2c_33_11599190a2c_33_11599192a2c_33_11599194a2c_33_11599196a2c_33_11599198a2c_33_11599200a2c_33_11599202a2c_33_11599204*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:::ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_call_115991572 
a2c_33/StatefulPartitionedCall
IdentityIdentity'a2c_33/StatefulPartitionedCall:output:0^a2c_33/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity'a2c_33/StatefulPartitionedCall:output:1^a2c_33/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1 

Identity_2Identity'a2c_33/StatefulPartitionedCall:output:2^a2c_33/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::2@
a2c_33/StatefulPartitionedCalla2c_33/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
±
´
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_11599335

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó

4__inference_Policy_mu_readout_layer_call_fn_11599859

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_115994692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í

1__inference_Policy_sigma_1_layer_call_fn_11599997

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_115993352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
³
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_11600068

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±
´
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_11600008

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
³
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_11600028

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
´
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_11599308

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µe

$__inference__traced_restore_11600256
file_prefix-
)assignvariableop_policy_mu_readout_kernel-
)assignvariableop_1_policy_mu_readout_bias2
.assignvariableop_2_policy_sigma_readout_kernel0
,assignvariableop_3_policy_sigma_readout_bias+
'assignvariableop_4_value_readout_kernel)
%assignvariableop_5_value_readout_bias)
%assignvariableop_6_policy_mu_0_kernel'
#assignvariableop_7_policy_mu_0_bias)
%assignvariableop_8_policy_mu_1_kernel'
#assignvariableop_9_policy_mu_1_bias*
&assignvariableop_10_policy_mu_2_kernel(
$assignvariableop_11_policy_mu_2_bias-
)assignvariableop_12_policy_sigma_0_kernel+
'assignvariableop_13_policy_sigma_0_bias-
)assignvariableop_14_policy_sigma_1_kernel+
'assignvariableop_15_policy_sigma_1_bias-
)assignvariableop_16_policy_sigma_2_kernel+
'assignvariableop_17_policy_sigma_2_bias,
(assignvariableop_18_value_layer_0_kernel*
&assignvariableop_19_value_layer_0_bias,
(assignvariableop_20_value_layer_1_kernel*
&assignvariableop_21_value_layer_1_bias,
(assignvariableop_22_value_layer_2_kernel*
&assignvariableop_23_value_layer_2_bias
identity_25¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ï

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û	
valueÑ	BÎ	B,readout_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB*readout_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_sigma/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_sigma/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¨
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¨
AssignVariableOpAssignVariableOp)assignvariableop_policy_mu_readout_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_policy_mu_readout_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_policy_sigma_readout_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_policy_sigma_readout_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_value_readout_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ª
AssignVariableOp_5AssignVariableOp%assignvariableop_5_value_readout_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_policy_mu_0_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¨
AssignVariableOp_7AssignVariableOp#assignvariableop_7_policy_mu_0_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_policy_mu_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¨
AssignVariableOp_9AssignVariableOp#assignvariableop_9_policy_mu_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_policy_mu_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¬
AssignVariableOp_11AssignVariableOp$assignvariableop_11_policy_mu_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_policy_sigma_0_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¯
AssignVariableOp_13AssignVariableOp'assignvariableop_13_policy_sigma_0_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14±
AssignVariableOp_14AssignVariableOp)assignvariableop_14_policy_sigma_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¯
AssignVariableOp_15AssignVariableOp'assignvariableop_15_policy_sigma_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_policy_sigma_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¯
AssignVariableOp_17AssignVariableOp'assignvariableop_17_policy_sigma_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp(assignvariableop_18_value_layer_0_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp&assignvariableop_19_value_layer_0_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_value_layer_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp&assignvariableop_21_value_layer_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_value_layer_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23®
AssignVariableOp_23AssignVariableOp&assignvariableop_23_value_layer_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpî
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24á
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
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
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
ð`
©

__inference_call_11599157
input_state.
*policy_mu_0_matmul_readvariableop_resource/
+policy_mu_0_biasadd_readvariableop_resource.
*policy_mu_1_matmul_readvariableop_resource/
+policy_mu_1_biasadd_readvariableop_resource.
*policy_mu_2_matmul_readvariableop_resource/
+policy_mu_2_biasadd_readvariableop_resource1
-policy_sigma_0_matmul_readvariableop_resource2
.policy_sigma_0_biasadd_readvariableop_resource1
-policy_sigma_1_matmul_readvariableop_resource2
.policy_sigma_1_biasadd_readvariableop_resource1
-policy_sigma_2_matmul_readvariableop_resource2
.policy_sigma_2_biasadd_readvariableop_resource0
,value_layer_0_matmul_readvariableop_resource1
-value_layer_0_biasadd_readvariableop_resource0
,value_layer_1_matmul_readvariableop_resource1
-value_layer_1_biasadd_readvariableop_resource0
,value_layer_2_matmul_readvariableop_resource1
-value_layer_2_biasadd_readvariableop_resource4
0policy_mu_readout_matmul_readvariableop_resource5
1policy_mu_readout_biasadd_readvariableop_resource7
3policy_sigma_readout_matmul_readvariableop_resource8
4policy_sigma_readout_biasadd_readvariableop_resource0
,value_readout_matmul_readvariableop_resource1
-value_readout_biasadd_readvariableop_resource
identity

identity_1

identity_2±
!Policy_mu_0/MatMul/ReadVariableOpReadVariableOp*policy_mu_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!Policy_mu_0/MatMul/ReadVariableOp
Policy_mu_0/MatMulMatMulinput_state)Policy_mu_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/MatMul°
"Policy_mu_0/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_0/BiasAdd/ReadVariableOp±
Policy_mu_0/BiasAddBiasAddPolicy_mu_0/MatMul:product:0*Policy_mu_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/BiasAdd|
Policy_mu_0/ReluReluPolicy_mu_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/Relu±
!Policy_mu_1/MatMul/ReadVariableOpReadVariableOp*policy_mu_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_1/MatMul/ReadVariableOp¯
Policy_mu_1/MatMulMatMulPolicy_mu_0/Relu:activations:0)Policy_mu_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/MatMul°
"Policy_mu_1/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_1/BiasAdd/ReadVariableOp±
Policy_mu_1/BiasAddBiasAddPolicy_mu_1/MatMul:product:0*Policy_mu_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/BiasAdd|
Policy_mu_1/ReluReluPolicy_mu_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/Relu±
!Policy_mu_2/MatMul/ReadVariableOpReadVariableOp*policy_mu_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_2/MatMul/ReadVariableOp¯
Policy_mu_2/MatMulMatMulPolicy_mu_1/Relu:activations:0)Policy_mu_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/MatMul°
"Policy_mu_2/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_2/BiasAdd/ReadVariableOp±
Policy_mu_2/BiasAddBiasAddPolicy_mu_2/MatMul:product:0*Policy_mu_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/BiasAdd|
Policy_mu_2/ReluReluPolicy_mu_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/Reluº
$Policy_sigma_0/MatMul/ReadVariableOpReadVariableOp-policy_sigma_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Policy_sigma_0/MatMul/ReadVariableOp¥
Policy_sigma_0/MatMulMatMulinput_state,Policy_sigma_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/MatMul¹
%Policy_sigma_0/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_0/BiasAdd/ReadVariableOp½
Policy_sigma_0/BiasAddBiasAddPolicy_sigma_0/MatMul:product:0-Policy_sigma_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/BiasAdd
Policy_sigma_0/ReluReluPolicy_sigma_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/Reluº
$Policy_sigma_1/MatMul/ReadVariableOpReadVariableOp-policy_sigma_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_1/MatMul/ReadVariableOp»
Policy_sigma_1/MatMulMatMul!Policy_sigma_0/Relu:activations:0,Policy_sigma_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/MatMul¹
%Policy_sigma_1/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_1/BiasAdd/ReadVariableOp½
Policy_sigma_1/BiasAddBiasAddPolicy_sigma_1/MatMul:product:0-Policy_sigma_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/BiasAdd
Policy_sigma_1/ReluReluPolicy_sigma_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/Reluº
$Policy_sigma_2/MatMul/ReadVariableOpReadVariableOp-policy_sigma_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_2/MatMul/ReadVariableOp»
Policy_sigma_2/MatMulMatMul!Policy_sigma_1/Relu:activations:0,Policy_sigma_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/MatMul¹
%Policy_sigma_2/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_2/BiasAdd/ReadVariableOp½
Policy_sigma_2/BiasAddBiasAddPolicy_sigma_2/MatMul:product:0-Policy_sigma_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/BiasAdd
Policy_sigma_2/ReluReluPolicy_sigma_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/Relu·
#Value_layer_0/MatMul/ReadVariableOpReadVariableOp,value_layer_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_layer_0/MatMul/ReadVariableOp¢
Value_layer_0/MatMulMatMulinput_state+Value_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/MatMul¶
$Value_layer_0/BiasAdd/ReadVariableOpReadVariableOp-value_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_0/BiasAdd/ReadVariableOp¹
Value_layer_0/BiasAddBiasAddValue_layer_0/MatMul:product:0,Value_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/BiasAdd
Value_layer_0/ReluReluValue_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/Relu·
#Value_layer_1/MatMul/ReadVariableOpReadVariableOp,value_layer_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_1/MatMul/ReadVariableOp·
Value_layer_1/MatMulMatMul Value_layer_0/Relu:activations:0+Value_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/MatMul¶
$Value_layer_1/BiasAdd/ReadVariableOpReadVariableOp-value_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_1/BiasAdd/ReadVariableOp¹
Value_layer_1/BiasAddBiasAddValue_layer_1/MatMul:product:0,Value_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/BiasAdd
Value_layer_1/ReluReluValue_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/Relu·
#Value_layer_2/MatMul/ReadVariableOpReadVariableOp,value_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_2/MatMul/ReadVariableOp·
Value_layer_2/MatMulMatMul Value_layer_1/Relu:activations:0+Value_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/MatMul¶
$Value_layer_2/BiasAdd/ReadVariableOpReadVariableOp-value_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_2/BiasAdd/ReadVariableOp¹
Value_layer_2/BiasAddBiasAddValue_layer_2/MatMul:product:0,Value_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/BiasAdd
Value_layer_2/ReluReluValue_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/ReluÃ
'Policy_mu_readout/MatMul/ReadVariableOpReadVariableOp0policy_mu_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'Policy_mu_readout/MatMul/ReadVariableOpÁ
Policy_mu_readout/MatMulMatMulPolicy_mu_2/Relu:activations:0/Policy_mu_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/MatMulÂ
(Policy_mu_readout/BiasAdd/ReadVariableOpReadVariableOp1policy_mu_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Policy_mu_readout/BiasAdd/ReadVariableOpÉ
Policy_mu_readout/BiasAddBiasAdd"Policy_mu_readout/MatMul:product:00Policy_mu_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/BiasAddd
SqueezeSqueeze"Policy_mu_readout/BiasAdd:output:0*
T0*
_output_shapes
:2	
SqueezeÌ
*Policy_sigma_readout/MatMul/ReadVariableOpReadVariableOp3policy_sigma_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*Policy_sigma_readout/MatMul/ReadVariableOpÍ
Policy_sigma_readout/MatMulMatMul!Policy_sigma_2/Relu:activations:02Policy_sigma_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/MatMulË
+Policy_sigma_readout/BiasAdd/ReadVariableOpReadVariableOp4policy_sigma_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Policy_sigma_readout/BiasAdd/ReadVariableOpÕ
Policy_sigma_readout/BiasAddBiasAdd%Policy_sigma_readout/MatMul:product:03Policy_sigma_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/BiasAddj
AbsAbs%Policy_sigma_readout/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
AbsM
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1·
#Value_readout/MatMul/ReadVariableOpReadVariableOp,value_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_readout/MatMul/ReadVariableOp·
Value_readout/MatMulMatMul Value_layer_2/Relu:activations:0+Value_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/MatMul¶
$Value_readout/BiasAdd/ReadVariableOpReadVariableOp-value_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Value_readout/BiasAdd/ReadVariableOp¹
Value_readout/BiasAddBiasAddValue_readout/MatMul:product:0,Value_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/BiasAddU
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes
:2

Identity[

Identity_1IdentitySqueeze_1:output:0*
T0*
_output_shapes
:2

Identity_1v

Identity_2IdentityValue_readout/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_state
¤^
©

__inference_call_11599750
input_state.
*policy_mu_0_matmul_readvariableop_resource/
+policy_mu_0_biasadd_readvariableop_resource.
*policy_mu_1_matmul_readvariableop_resource/
+policy_mu_1_biasadd_readvariableop_resource.
*policy_mu_2_matmul_readvariableop_resource/
+policy_mu_2_biasadd_readvariableop_resource1
-policy_sigma_0_matmul_readvariableop_resource2
.policy_sigma_0_biasadd_readvariableop_resource1
-policy_sigma_1_matmul_readvariableop_resource2
.policy_sigma_1_biasadd_readvariableop_resource1
-policy_sigma_2_matmul_readvariableop_resource2
.policy_sigma_2_biasadd_readvariableop_resource0
,value_layer_0_matmul_readvariableop_resource1
-value_layer_0_biasadd_readvariableop_resource0
,value_layer_1_matmul_readvariableop_resource1
-value_layer_1_biasadd_readvariableop_resource0
,value_layer_2_matmul_readvariableop_resource1
-value_layer_2_biasadd_readvariableop_resource4
0policy_mu_readout_matmul_readvariableop_resource5
1policy_mu_readout_biasadd_readvariableop_resource7
3policy_sigma_readout_matmul_readvariableop_resource8
4policy_sigma_readout_biasadd_readvariableop_resource0
,value_readout_matmul_readvariableop_resource1
-value_readout_biasadd_readvariableop_resource
identity

identity_1

identity_2±
!Policy_mu_0/MatMul/ReadVariableOpReadVariableOp*policy_mu_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!Policy_mu_0/MatMul/ReadVariableOp
Policy_mu_0/MatMulMatMulinput_state)Policy_mu_0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_0/MatMul°
"Policy_mu_0/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_0/BiasAdd/ReadVariableOp¨
Policy_mu_0/BiasAddBiasAddPolicy_mu_0/MatMul:product:0*Policy_mu_0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_0/BiasAdds
Policy_mu_0/ReluReluPolicy_mu_0/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_mu_0/Relu±
!Policy_mu_1/MatMul/ReadVariableOpReadVariableOp*policy_mu_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_1/MatMul/ReadVariableOp¦
Policy_mu_1/MatMulMatMulPolicy_mu_0/Relu:activations:0)Policy_mu_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_1/MatMul°
"Policy_mu_1/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_1/BiasAdd/ReadVariableOp¨
Policy_mu_1/BiasAddBiasAddPolicy_mu_1/MatMul:product:0*Policy_mu_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_1/BiasAdds
Policy_mu_1/ReluReluPolicy_mu_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_mu_1/Relu±
!Policy_mu_2/MatMul/ReadVariableOpReadVariableOp*policy_mu_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_2/MatMul/ReadVariableOp¦
Policy_mu_2/MatMulMatMulPolicy_mu_1/Relu:activations:0)Policy_mu_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_2/MatMul°
"Policy_mu_2/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_2/BiasAdd/ReadVariableOp¨
Policy_mu_2/BiasAddBiasAddPolicy_mu_2/MatMul:product:0*Policy_mu_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_2/BiasAdds
Policy_mu_2/ReluReluPolicy_mu_2/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_mu_2/Reluº
$Policy_sigma_0/MatMul/ReadVariableOpReadVariableOp-policy_sigma_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Policy_sigma_0/MatMul/ReadVariableOp
Policy_sigma_0/MatMulMatMulinput_state,Policy_sigma_0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_0/MatMul¹
%Policy_sigma_0/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_0/BiasAdd/ReadVariableOp´
Policy_sigma_0/BiasAddBiasAddPolicy_sigma_0/MatMul:product:0-Policy_sigma_0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_0/BiasAdd|
Policy_sigma_0/ReluReluPolicy_sigma_0/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_sigma_0/Reluº
$Policy_sigma_1/MatMul/ReadVariableOpReadVariableOp-policy_sigma_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_1/MatMul/ReadVariableOp²
Policy_sigma_1/MatMulMatMul!Policy_sigma_0/Relu:activations:0,Policy_sigma_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_1/MatMul¹
%Policy_sigma_1/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_1/BiasAdd/ReadVariableOp´
Policy_sigma_1/BiasAddBiasAddPolicy_sigma_1/MatMul:product:0-Policy_sigma_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_1/BiasAdd|
Policy_sigma_1/ReluReluPolicy_sigma_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_sigma_1/Reluº
$Policy_sigma_2/MatMul/ReadVariableOpReadVariableOp-policy_sigma_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_2/MatMul/ReadVariableOp²
Policy_sigma_2/MatMulMatMul!Policy_sigma_1/Relu:activations:0,Policy_sigma_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_2/MatMul¹
%Policy_sigma_2/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_2/BiasAdd/ReadVariableOp´
Policy_sigma_2/BiasAddBiasAddPolicy_sigma_2/MatMul:product:0-Policy_sigma_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_2/BiasAdd|
Policy_sigma_2/ReluReluPolicy_sigma_2/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_sigma_2/Relu·
#Value_layer_0/MatMul/ReadVariableOpReadVariableOp,value_layer_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_layer_0/MatMul/ReadVariableOp
Value_layer_0/MatMulMatMulinput_state+Value_layer_0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_0/MatMul¶
$Value_layer_0/BiasAdd/ReadVariableOpReadVariableOp-value_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_0/BiasAdd/ReadVariableOp°
Value_layer_0/BiasAddBiasAddValue_layer_0/MatMul:product:0,Value_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_0/BiasAddy
Value_layer_0/ReluReluValue_layer_0/BiasAdd:output:0*
T0*
_output_shapes

: 2
Value_layer_0/Relu·
#Value_layer_1/MatMul/ReadVariableOpReadVariableOp,value_layer_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_1/MatMul/ReadVariableOp®
Value_layer_1/MatMulMatMul Value_layer_0/Relu:activations:0+Value_layer_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_1/MatMul¶
$Value_layer_1/BiasAdd/ReadVariableOpReadVariableOp-value_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_1/BiasAdd/ReadVariableOp°
Value_layer_1/BiasAddBiasAddValue_layer_1/MatMul:product:0,Value_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_1/BiasAddy
Value_layer_1/ReluReluValue_layer_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
Value_layer_1/Relu·
#Value_layer_2/MatMul/ReadVariableOpReadVariableOp,value_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_2/MatMul/ReadVariableOp®
Value_layer_2/MatMulMatMul Value_layer_1/Relu:activations:0+Value_layer_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_2/MatMul¶
$Value_layer_2/BiasAdd/ReadVariableOpReadVariableOp-value_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_2/BiasAdd/ReadVariableOp°
Value_layer_2/BiasAddBiasAddValue_layer_2/MatMul:product:0,Value_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_2/BiasAddy
Value_layer_2/ReluReluValue_layer_2/BiasAdd:output:0*
T0*
_output_shapes

: 2
Value_layer_2/ReluÃ
'Policy_mu_readout/MatMul/ReadVariableOpReadVariableOp0policy_mu_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'Policy_mu_readout/MatMul/ReadVariableOp¸
Policy_mu_readout/MatMulMatMulPolicy_mu_2/Relu:activations:0/Policy_mu_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_mu_readout/MatMulÂ
(Policy_mu_readout/BiasAdd/ReadVariableOpReadVariableOp1policy_mu_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Policy_mu_readout/BiasAdd/ReadVariableOpÀ
Policy_mu_readout/BiasAddBiasAdd"Policy_mu_readout/MatMul:product:00Policy_mu_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_mu_readout/BiasAddf
SqueezeSqueeze"Policy_mu_readout/BiasAdd:output:0*
T0*
_output_shapes
:2	
SqueezeÌ
*Policy_sigma_readout/MatMul/ReadVariableOpReadVariableOp3policy_sigma_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*Policy_sigma_readout/MatMul/ReadVariableOpÄ
Policy_sigma_readout/MatMulMatMul!Policy_sigma_2/Relu:activations:02Policy_sigma_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_sigma_readout/MatMulË
+Policy_sigma_readout/BiasAdd/ReadVariableOpReadVariableOp4policy_sigma_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Policy_sigma_readout/BiasAdd/ReadVariableOpÌ
Policy_sigma_readout/BiasAddBiasAdd%Policy_sigma_readout/MatMul:product:03Policy_sigma_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_sigma_readout/BiasAdda
AbsAbs%Policy_sigma_readout/BiasAdd:output:0*
T0*
_output_shapes

:2
AbsO
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1·
#Value_readout/MatMul/ReadVariableOpReadVariableOp,value_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_readout/MatMul/ReadVariableOp®
Value_readout/MatMulMatMul Value_layer_2/Relu:activations:0+Value_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Value_readout/MatMul¶
$Value_readout/BiasAdd/ReadVariableOpReadVariableOp-value_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Value_readout/BiasAdd/ReadVariableOp°
Value_readout/BiasAddBiasAddValue_readout/MatMul:product:0,Value_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Value_readout/BiasAddW
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes
:2

Identity]

Identity_1IdentitySqueeze_1:output:0*
T0*
_output_shapes
:2

Identity_1m

Identity_2IdentityValue_readout/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*}
_input_shapesl
j::::::::::::::::::::::::::K G

_output_shapes

:
%
_user_specified_nameinput_state
°
³
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_11600048

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
8
Þ

!__inference__traced_save_11600174
file_prefix7
3savev2_policy_mu_readout_kernel_read_readvariableop5
1savev2_policy_mu_readout_bias_read_readvariableop:
6savev2_policy_sigma_readout_kernel_read_readvariableop8
4savev2_policy_sigma_readout_bias_read_readvariableop3
/savev2_value_readout_kernel_read_readvariableop1
-savev2_value_readout_bias_read_readvariableop1
-savev2_policy_mu_0_kernel_read_readvariableop/
+savev2_policy_mu_0_bias_read_readvariableop1
-savev2_policy_mu_1_kernel_read_readvariableop/
+savev2_policy_mu_1_bias_read_readvariableop1
-savev2_policy_mu_2_kernel_read_readvariableop/
+savev2_policy_mu_2_bias_read_readvariableop4
0savev2_policy_sigma_0_kernel_read_readvariableop2
.savev2_policy_sigma_0_bias_read_readvariableop4
0savev2_policy_sigma_1_kernel_read_readvariableop2
.savev2_policy_sigma_1_bias_read_readvariableop4
0savev2_policy_sigma_2_kernel_read_readvariableop2
.savev2_policy_sigma_2_bias_read_readvariableop3
/savev2_value_layer_0_kernel_read_readvariableop1
-savev2_value_layer_0_bias_read_readvariableop3
/savev2_value_layer_1_kernel_read_readvariableop1
-savev2_value_layer_1_bias_read_readvariableop3
/savev2_value_layer_2_kernel_read_readvariableop1
-savev2_value_layer_2_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5983aeadde8a43de9f1715eb05a490d8/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÉ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û	
valueÑ	BÎ	B,readout_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB*readout_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_sigma/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_sigma/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesº
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesâ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_policy_mu_readout_kernel_read_readvariableop1savev2_policy_mu_readout_bias_read_readvariableop6savev2_policy_sigma_readout_kernel_read_readvariableop4savev2_policy_sigma_readout_bias_read_readvariableop/savev2_value_readout_kernel_read_readvariableop-savev2_value_readout_bias_read_readvariableop-savev2_policy_mu_0_kernel_read_readvariableop+savev2_policy_mu_0_bias_read_readvariableop-savev2_policy_mu_1_kernel_read_readvariableop+savev2_policy_mu_1_bias_read_readvariableop-savev2_policy_mu_2_kernel_read_readvariableop+savev2_policy_mu_2_bias_read_readvariableop0savev2_policy_sigma_0_kernel_read_readvariableop.savev2_policy_sigma_0_bias_read_readvariableop0savev2_policy_sigma_1_kernel_read_readvariableop.savev2_policy_sigma_1_bias_read_readvariableop0savev2_policy_sigma_2_kernel_read_readvariableop.savev2_policy_sigma_2_bias_read_readvariableop/savev2_value_layer_0_kernel_read_readvariableop-savev2_value_layer_0_bias_read_readvariableop/savev2_value_layer_1_kernel_read_readvariableop-savev2_value_layer_1_bias_read_readvariableop/savev2_value_layer_2_kernel_read_readvariableop-savev2_value_layer_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ù
_input_shapesÇ
Ä: : :: :: :: : :  : :  : : : :  : :  : : : :  : :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: 
°
³
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_11599443

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ø
·
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_11599469

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç

.__inference_Policy_mu_2_layer_call_fn_11599957

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_115992812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_Value_readout_layer_call_fn_11599897

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_readout_layer_call_and_return_conditional_losses_115995242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
±
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_11599908

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
³
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_11599416

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ô
³
K__inference_Value_readout_layer_call_and_return_conditional_losses_11599888

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ð`
©

__inference_call_11599840
input_state.
*policy_mu_0_matmul_readvariableop_resource/
+policy_mu_0_biasadd_readvariableop_resource.
*policy_mu_1_matmul_readvariableop_resource/
+policy_mu_1_biasadd_readvariableop_resource.
*policy_mu_2_matmul_readvariableop_resource/
+policy_mu_2_biasadd_readvariableop_resource1
-policy_sigma_0_matmul_readvariableop_resource2
.policy_sigma_0_biasadd_readvariableop_resource1
-policy_sigma_1_matmul_readvariableop_resource2
.policy_sigma_1_biasadd_readvariableop_resource1
-policy_sigma_2_matmul_readvariableop_resource2
.policy_sigma_2_biasadd_readvariableop_resource0
,value_layer_0_matmul_readvariableop_resource1
-value_layer_0_biasadd_readvariableop_resource0
,value_layer_1_matmul_readvariableop_resource1
-value_layer_1_biasadd_readvariableop_resource0
,value_layer_2_matmul_readvariableop_resource1
-value_layer_2_biasadd_readvariableop_resource4
0policy_mu_readout_matmul_readvariableop_resource5
1policy_mu_readout_biasadd_readvariableop_resource7
3policy_sigma_readout_matmul_readvariableop_resource8
4policy_sigma_readout_biasadd_readvariableop_resource0
,value_readout_matmul_readvariableop_resource1
-value_readout_biasadd_readvariableop_resource
identity

identity_1

identity_2±
!Policy_mu_0/MatMul/ReadVariableOpReadVariableOp*policy_mu_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!Policy_mu_0/MatMul/ReadVariableOp
Policy_mu_0/MatMulMatMulinput_state)Policy_mu_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/MatMul°
"Policy_mu_0/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_0/BiasAdd/ReadVariableOp±
Policy_mu_0/BiasAddBiasAddPolicy_mu_0/MatMul:product:0*Policy_mu_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/BiasAdd|
Policy_mu_0/ReluReluPolicy_mu_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/Relu±
!Policy_mu_1/MatMul/ReadVariableOpReadVariableOp*policy_mu_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_1/MatMul/ReadVariableOp¯
Policy_mu_1/MatMulMatMulPolicy_mu_0/Relu:activations:0)Policy_mu_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/MatMul°
"Policy_mu_1/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_1/BiasAdd/ReadVariableOp±
Policy_mu_1/BiasAddBiasAddPolicy_mu_1/MatMul:product:0*Policy_mu_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/BiasAdd|
Policy_mu_1/ReluReluPolicy_mu_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/Relu±
!Policy_mu_2/MatMul/ReadVariableOpReadVariableOp*policy_mu_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_2/MatMul/ReadVariableOp¯
Policy_mu_2/MatMulMatMulPolicy_mu_1/Relu:activations:0)Policy_mu_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/MatMul°
"Policy_mu_2/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_2/BiasAdd/ReadVariableOp±
Policy_mu_2/BiasAddBiasAddPolicy_mu_2/MatMul:product:0*Policy_mu_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/BiasAdd|
Policy_mu_2/ReluReluPolicy_mu_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/Reluº
$Policy_sigma_0/MatMul/ReadVariableOpReadVariableOp-policy_sigma_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Policy_sigma_0/MatMul/ReadVariableOp¥
Policy_sigma_0/MatMulMatMulinput_state,Policy_sigma_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/MatMul¹
%Policy_sigma_0/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_0/BiasAdd/ReadVariableOp½
Policy_sigma_0/BiasAddBiasAddPolicy_sigma_0/MatMul:product:0-Policy_sigma_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/BiasAdd
Policy_sigma_0/ReluReluPolicy_sigma_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/Reluº
$Policy_sigma_1/MatMul/ReadVariableOpReadVariableOp-policy_sigma_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_1/MatMul/ReadVariableOp»
Policy_sigma_1/MatMulMatMul!Policy_sigma_0/Relu:activations:0,Policy_sigma_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/MatMul¹
%Policy_sigma_1/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_1/BiasAdd/ReadVariableOp½
Policy_sigma_1/BiasAddBiasAddPolicy_sigma_1/MatMul:product:0-Policy_sigma_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/BiasAdd
Policy_sigma_1/ReluReluPolicy_sigma_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/Reluº
$Policy_sigma_2/MatMul/ReadVariableOpReadVariableOp-policy_sigma_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_2/MatMul/ReadVariableOp»
Policy_sigma_2/MatMulMatMul!Policy_sigma_1/Relu:activations:0,Policy_sigma_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/MatMul¹
%Policy_sigma_2/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_2/BiasAdd/ReadVariableOp½
Policy_sigma_2/BiasAddBiasAddPolicy_sigma_2/MatMul:product:0-Policy_sigma_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/BiasAdd
Policy_sigma_2/ReluReluPolicy_sigma_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/Relu·
#Value_layer_0/MatMul/ReadVariableOpReadVariableOp,value_layer_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_layer_0/MatMul/ReadVariableOp¢
Value_layer_0/MatMulMatMulinput_state+Value_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/MatMul¶
$Value_layer_0/BiasAdd/ReadVariableOpReadVariableOp-value_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_0/BiasAdd/ReadVariableOp¹
Value_layer_0/BiasAddBiasAddValue_layer_0/MatMul:product:0,Value_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/BiasAdd
Value_layer_0/ReluReluValue_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/Relu·
#Value_layer_1/MatMul/ReadVariableOpReadVariableOp,value_layer_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_1/MatMul/ReadVariableOp·
Value_layer_1/MatMulMatMul Value_layer_0/Relu:activations:0+Value_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/MatMul¶
$Value_layer_1/BiasAdd/ReadVariableOpReadVariableOp-value_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_1/BiasAdd/ReadVariableOp¹
Value_layer_1/BiasAddBiasAddValue_layer_1/MatMul:product:0,Value_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/BiasAdd
Value_layer_1/ReluReluValue_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/Relu·
#Value_layer_2/MatMul/ReadVariableOpReadVariableOp,value_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_2/MatMul/ReadVariableOp·
Value_layer_2/MatMulMatMul Value_layer_1/Relu:activations:0+Value_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/MatMul¶
$Value_layer_2/BiasAdd/ReadVariableOpReadVariableOp-value_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_2/BiasAdd/ReadVariableOp¹
Value_layer_2/BiasAddBiasAddValue_layer_2/MatMul:product:0,Value_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/BiasAdd
Value_layer_2/ReluReluValue_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/ReluÃ
'Policy_mu_readout/MatMul/ReadVariableOpReadVariableOp0policy_mu_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'Policy_mu_readout/MatMul/ReadVariableOpÁ
Policy_mu_readout/MatMulMatMulPolicy_mu_2/Relu:activations:0/Policy_mu_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/MatMulÂ
(Policy_mu_readout/BiasAdd/ReadVariableOpReadVariableOp1policy_mu_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Policy_mu_readout/BiasAdd/ReadVariableOpÉ
Policy_mu_readout/BiasAddBiasAdd"Policy_mu_readout/MatMul:product:00Policy_mu_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/BiasAddd
SqueezeSqueeze"Policy_mu_readout/BiasAdd:output:0*
T0*
_output_shapes
:2	
SqueezeÌ
*Policy_sigma_readout/MatMul/ReadVariableOpReadVariableOp3policy_sigma_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*Policy_sigma_readout/MatMul/ReadVariableOpÍ
Policy_sigma_readout/MatMulMatMul!Policy_sigma_2/Relu:activations:02Policy_sigma_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/MatMulË
+Policy_sigma_readout/BiasAdd/ReadVariableOpReadVariableOp4policy_sigma_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Policy_sigma_readout/BiasAdd/ReadVariableOpÕ
Policy_sigma_readout/BiasAddBiasAdd%Policy_sigma_readout/MatMul:product:03Policy_sigma_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/BiasAddj
AbsAbs%Policy_sigma_readout/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
AbsM
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1·
#Value_readout/MatMul/ReadVariableOpReadVariableOp,value_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_readout/MatMul/ReadVariableOp·
Value_readout/MatMulMatMul Value_layer_2/Relu:activations:0+Value_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/MatMul¶
$Value_readout/BiasAdd/ReadVariableOpReadVariableOp-value_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Value_readout/BiasAdd/ReadVariableOp¹
Value_readout/BiasAddBiasAddValue_readout/MatMul:product:0,Value_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/BiasAddU
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes
:2

Identity[

Identity_1IdentitySqueeze_1:output:0*
T0*
_output_shapes
:2

Identity_1v

Identity_2IdentityValue_readout/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_state
±
´
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_11599988

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_Value_layer_0_layer_call_fn_11600037

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_115993892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
º
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_11599496

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û
º
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_11599869

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
±
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_11599227

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ó
&__inference_signature_wrapper_11599660
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:::ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_115992122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
®
±
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_11599281

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±
´
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_11599362

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultò
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ'
mu!
StatefulPartitionedCall:0*
sigma!
StatefulPartitionedCall:1B
value_estimate0
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ð

mu_layer

readout_mu
sigma_layer
readout_sigma
value_layer
readout_value
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
_default_save_signature
+&call_and_return_all_conditional_losses
 __call__
	¡call"î
_tf_keras_modelÔ{"class_name": "A2C", "name": "a2c_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "A2C"}}
5
0
1
2"
trackable_list_wrapper


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"Ý
_tf_keras_layerÃ{"class_name": "Dense", "name": "Policy_mu_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_readout", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
5
0
1
2"
trackable_list_wrapper


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"ã
_tf_keras_layerÉ{"class_name": "Dense", "name": "Policy_sigma_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_readout", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
5
0
1
 2"
trackable_list_wrapper
ü

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"Õ
_tf_keras_layer»{"class_name": "Dense", "name": "Value_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_readout", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
Ö
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23"
trackable_list_wrapper
Ö
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
9metrics
:non_trainable_variables
;layer_regularization_losses
<layer_metrics
trainable_variables
	variables
	regularization_losses

=layers
 __call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¨serving_default"
signature_map
÷

'kernel
(bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+©&call_and_return_all_conditional_losses
ª__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "Policy_mu_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_0", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 31}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 31]}}
÷

)kernel
*bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "Policy_mu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
÷

+kernel
,bias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+­&call_and_return_all_conditional_losses
®__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "Policy_mu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
*:( 2Policy_mu_readout/kernel
$:"2Policy_mu_readout/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Jmetrics
Knon_trainable_variables
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
	variables
regularization_losses

Nlayers
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
ý

-kernel
.bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "Policy_sigma_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_0", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 31}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 31]}}
ý

/kernel
0bias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
+±&call_and_return_all_conditional_losses
²__call__"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "Policy_sigma_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
ý

1kernel
2bias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
+³&call_and_return_all_conditional_losses
´__call__"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "Policy_sigma_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
-:+ 2Policy_sigma_readout/kernel
':%2Policy_sigma_readout/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
[metrics
\non_trainable_variables
]layer_metrics
^layer_regularization_losses
trainable_variables
	variables
regularization_losses

_layers
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
û

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Value_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_layer_0", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 31}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 31]}}
û

5kernel
6bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Value_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_layer_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
û

7kernel
8bias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Value_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
&:$ 2Value_readout/kernel
 :2Value_readout/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
lmetrics
mnon_trainable_variables
nlayer_metrics
olayer_regularization_losses
#trainable_variables
$	variables
%regularization_losses

players
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
$:" 2Policy_mu_0/kernel
: 2Policy_mu_0/bias
$:"  2Policy_mu_1/kernel
: 2Policy_mu_1/bias
$:"  2Policy_mu_2/kernel
: 2Policy_mu_2/bias
':% 2Policy_sigma_0/kernel
!: 2Policy_sigma_0/bias
':%  2Policy_sigma_1/kernel
!: 2Policy_sigma_1/bias
':%  2Policy_sigma_2/kernel
!: 2Policy_sigma_2/bias
&:$ 2Value_layer_0/kernel
 : 2Value_layer_0/bias
&:$  2Value_layer_1/kernel
 : 2Value_layer_1/bias
&:$  2Value_layer_2/kernel
 : 2Value_layer_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
v
0
1
2
3
4
5
6
7
8
9
 10
11"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
qmetrics
rnon_trainable_variables
slayer_metrics
tlayer_regularization_losses
>trainable_variables
?	variables
@regularization_losses

ulayers
ª__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
vmetrics
wnon_trainable_variables
xlayer_metrics
ylayer_regularization_losses
Btrainable_variables
C	variables
Dregularization_losses

zlayers
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
{metrics
|non_trainable_variables
}layer_metrics
~layer_regularization_losses
Ftrainable_variables
G	variables
Hregularization_losses

layers
®__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
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
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
Otrainable_variables
P	variables
Qregularization_losses
layers
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
Strainable_variables
T	variables
Uregularization_losses
layers
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
Wtrainable_variables
X	variables
Yregularization_losses
layers
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
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
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
`trainable_variables
a	variables
bregularization_losses
layers
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
dtrainable_variables
e	variables
fregularization_losses
layers
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
htrainable_variables
i	variables
jregularization_losses
layers
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
á2Þ
#__inference__wrapped_model_11599212¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
2
D__inference_a2c_33_layer_call_and_return_conditional_losses_11599543Ë
²
FullArgSpec"
args
jself
jinput_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ü2ù
)__inference_a2c_33_layer_call_fn_11599601Ë
²
FullArgSpec"
args
jself
jinput_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ã2à
__inference_call_11599750
__inference_call_11599840§
²
FullArgSpec"
args
jself
jinput_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ù2ö
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_11599850¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
4__inference_Policy_mu_readout_layer_call_fn_11599859¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ü2ù
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_11599869¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
á2Þ
7__inference_Policy_sigma_readout_layer_call_fn_11599878¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_Value_readout_layer_call_and_return_conditional_losses_11599888¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_Value_readout_layer_call_fn_11599897¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5B3
&__inference_signature_wrapper_11599660input_1
ó2ð
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_11599908¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_Policy_mu_0_layer_call_fn_11599917¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_11599928¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_Policy_mu_1_layer_call_fn_11599937¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_11599948¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_Policy_mu_2_layer_call_fn_11599957¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_11599968¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Policy_sigma_0_layer_call_fn_11599977¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_11599988¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Policy_sigma_1_layer_call_fn_11599997¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_11600008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Policy_sigma_2_layer_call_fn_11600017¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_11600028¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_Value_layer_0_layer_call_fn_11600037¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_11600048¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_Value_layer_1_layer_call_fn_11600057¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_11600068¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_Value_layer_2_layer_call_fn_11600077¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ©
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_11599908\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Policy_mu_0_layer_call_fn_11599917O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ ©
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_11599928\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Policy_mu_1_layer_call_fn_11599937O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ©
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_11599948\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Policy_mu_2_layer_call_fn_11599957O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¯
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_11599850\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_Policy_mu_readout_layer_call_fn_11599859O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¬
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_11599968\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
1__inference_Policy_sigma_0_layer_call_fn_11599977O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ ¬
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_11599988\/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
1__inference_Policy_sigma_1_layer_call_fn_11599997O/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¬
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_11600008\12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
1__inference_Policy_sigma_2_layer_call_fn_11600017O12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ²
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_11599869\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_Policy_sigma_readout_layer_call_fn_11599878O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ«
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_11600028\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_Value_layer_0_layer_call_fn_11600037O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ «
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_11600048\56/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_Value_layer_1_layer_call_fn_11600057O56/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ «
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_11600068\78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_Value_layer_2_layer_call_fn_11600077O78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ «
K__inference_Value_readout_layer_call_and_return_conditional_losses_11599888\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_Value_readout_layer_call_fn_11599897O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿå
#__inference__wrapped_model_11599212½'()*+,-./012345678!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "oªl

mu

mu

sigma
sigma
:
value_estimate(%
value_estimateÿÿÿÿÿÿÿÿÿ
D__inference_a2c_33_layer_call_and_return_conditional_losses_11599543Í'()*+,-./012345678!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "¢|
uªr

mu
0/mu

sigma
0/sigma
<
value_estimate*'
0/value_estimateÿÿÿÿÿÿÿÿÿ
 ë
)__inference_a2c_33_layer_call_fn_11599601½'()*+,-./012345678!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "oªl

mu

mu

sigma
sigma
:
value_estimate(%
value_estimateÿÿÿÿÿÿÿÿÿÑ
__inference_call_11599750³'()*+,-./012345678!"+¢(
!¢

input_state
ª "jªg

mu
mu

sigma
sigma
1
value_estimate
value_estimateß
__inference_call_11599840Á'()*+,-./012345678!"4¢1
*¢'
%"
input_stateÿÿÿÿÿÿÿÿÿ
ª "oªl

mu

mu

sigma
sigma
:
value_estimate(%
value_estimateÿÿÿÿÿÿÿÿÿó
&__inference_signature_wrapper_11599660È'()*+,-./012345678!";¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"oªl

mu

mu

sigma
sigma
:
value_estimate(%
value_estimateÿÿÿÿÿÿÿÿÿ