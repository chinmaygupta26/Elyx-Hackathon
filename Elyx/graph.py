from graphviz import Digraph

def plot_elyx_graph():
    dot = Digraph(comment='Elyx Conversation Graph')

    # Nodes
    dot.node('START', 'START')
    dot.node('orchestrator', 'Ruby (Orchestrator)')
    dot.node('expert_consultation', 'Expert Consultation')
    dot.node('client_response', 'Client Response')
    dot.node('END', 'END')

    # Edges
    dot.edge('START', 'orchestrator')

    # Conditional routing from orchestrator
    dot.edge('orchestrator', 'expert_consultation', label='needs expert')
    dot.edge('orchestrator', 'client_response', label='handled by Ruby')

    # Expert always goes to client
    dot.edge('expert_consultation', 'client_response')

    # Client determines next step
    dot.edge('client_response', 'expert_consultation', label='continue with expert')
    dot.edge('client_response', 'orchestrator', label=' satisfied, back to Ruby')
    dot.edge('client_response', 'END', label=' conversation ends')

    # Render to file
    dot.render('elyx_graph', format='png', cleanup=True)
    print("Graph saved as elyx_graph.png")

# Call the function
plot_elyx_graph()
