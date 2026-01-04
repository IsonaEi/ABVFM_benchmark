import keypoint_moseq as kpms
import inspect

print("KPMS Version:", kpms.__version__)

print("\n--- extract_results signature ---")
try:
    print(inspect.signature(kpms.extract_results))
except Exception as e:
    print(e)
    # Check if docstring helps
    print(kpms.extract_results.__doc__)

print("\n--- expected_marginal_likelihoods signature ---")
try:
    print(inspect.signature(kpms.expected_marginal_likelihoods))
except Exception as e:
    print(e)
    
print("\n--- unbatch? ---")
# Check if unbatch is exposed or where it might be
if hasattr(kpms, 'unbatch'):
    print("kpms.unbatch found")
    print(inspect.signature(kpms.unbatch))
else:
    print("kpms.unbatch not found in top level")
