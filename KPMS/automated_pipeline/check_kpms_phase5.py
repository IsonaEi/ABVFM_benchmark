import keypoint_moseq as kpms
import inspect

print("KPMS Version:", kpms.__version__)

print("\n--- generate_transition_matrices signature ---")
try:
    print(inspect.signature(kpms.generate_transition_matrices))
except Exception as e:
    print(e)

print("\n--- visualize_transition_bigram signature ---")
try:
    print(inspect.signature(kpms.visualize_transition_bigram))
except Exception as e:
    print(e)

print("\n--- extract_results signature ---")
try:
    print(inspect.signature(kpms.extract_results))
except Exception as e:
    print(e)
