# Predicting Young's moduli with the averaging methods (Voigt, Reuss, Hill, and volume integral).

- averaging_methods_local_modulus_field: Computes the predicted Young's moduli from the local Young's modulus field E(r)=1/s_33(r). Strictly speaking, Voigt, Reuss, and Hill averages should be computed by averaging the stiffness and compliance tensors C and S according to the definition of these methods. However, for the Reuss approximation of the Young's modulus, averaging the local Young's modulus field is mathematically equivalent to the proper averaging of the compliance tensor.

- averaging_methods_stiffness_tensor: Computes the Voigt and Hill approximations for the Young's modulus by averaging the stiffness tensors, which is the mathematically correct way as opposed to averaging the local Young's modulus field.

The final predictions reported in the paper are from averaging_methods_local_modulus_field for the Reuss average and volume integral, and from averaging_methods_stiffness_tensor for the Voigt and Hill averages.
