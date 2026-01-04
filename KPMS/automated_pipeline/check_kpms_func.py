import keypoint_moseq as kpms
import inspect

print("\n--- plot_transition_graph signature ---")
try:
    if hasattr(kpms, 'plot_transition_graph'):
        print(inspect.signature(kpms.plot_transition_graph))
    else:
        print("plot_transition_graph NOT FOUND")
except Exception as e:
    print(e)
