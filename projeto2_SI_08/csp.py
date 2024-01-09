"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6).
changed by Sara C. Madeira - FCUL Sistemas Inteligentes 2016/17 and by Paulo Urbano 2018/19"""

from utils import argmin_random_tie, count, first
import search

from collections import defaultdict
from functools import reduce

import itertools
import re
import random


class CSP(search.Problem):

    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables        A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases. (For example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(N^4) for the
    explicit representation.) In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP.  Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        "Construct a CSP problem. If variables is empty, it becomes domains.keys()."
        variables = variables or list(domains.keys())

        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.initial = ()
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        "Add {var: val} to assignment; Discard the old value if any."
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        "Return the number of conflicts var=val has with other variables."
        # Subclasses may implement this more efficiently
        def conflict(var2):
            return (var2 in assignment and
                    not self.constraints(var, val, var2, assignment[var2]))
        return count(conflict(v) for v in self.neighbors[var])
    
    """   
    def nconflicts(self, var, val, assignment, verbose=False):
        "Return the number of conflicts var=val has with other variables."
        # Subclasses may implement this more efficiently
        def conflict(var2):
            return (var2 in assignment and
                    not self.constraints(var, val, var2, assignment[var2]))
        
        conta = 0
        for v in assignment and v in self.neighbors[var]:
            if verbose:
                print('Teste de conflito com', v)
            if conflict(v):
                if verbose:
                    print('Conflito')
                conta+=1
            elif verbose:
                print('Sem conflito!')
        return conta 
    """
    def display(self, assignment):
        "Show a human-readable representation of the CSP."
        # Subclasses can print in a prettier way, or display with a GUI
        print('CSP:', self, 'with assignment:', assignment)

    # These methods are for the tree- and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: nonconflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        "Perform an action and return the new state."
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        "The goal is to assign all variables, with all constraints satisfied."
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        "Start accumulating inferences from assuming var=value."
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        "Rule out var=value."
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        "Return all values for var that aren't currently ruled out."
        # if verbose:
        #    print('choice of ', var,'=',(self.curr_domains or self.domains)[var])
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        "Return the partial assignment implied by the current inferences."
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        "Undo a supposition and all inferences from it."
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        "Return a list of variables in current assignment that are in conflict"
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]

# ______________________________________________________________________________
# Constraint Propagation with AC-3


def AC3(csp, queue=None, removals=None, verbose = False):
    """[Figure 6.3]
    Sara C. Madeira - added verbose"""
    if queue is None:
        queue = [(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]]
       
    csp.support_pruning()
    if (verbose):
        print("----AC3")        
        print("----Set of Arcs to Check = ", queue)
        print("----Current Domains = ", csp.curr_domains)
    
    while queue:
        
        (Xi, Xj) = queue.pop()
        
        if (verbose):
            print ("-----Check", (Xi, Xj))
        
        if revise(csp, Xi, Xj, removals):
            if not csp.curr_domains[Xi]:
                if verbose:
                    print("-------Not consistent - Revise")        
                    print("-------Updated Current Domains = ", csp.curr_domains)
                    print("-------Updated Set of Arcs to Check = ", queue)
                return False
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj and (Xk,Xi) not in queue:
                    queue.append((Xk, Xi))
            
            if (verbose):
                print("-------Not consistent - Revise")        
                print("-------Updated Current Domains = ", csp.curr_domains)
                print("-------Updated Set of Arcs to Check = ", queue)
        elif(verbose):
                print("-------Consistent") 
            
    if (verbose):
            print("----Final Current Domains = ", csp.curr_domains)
            
    #return True
    
    return csp # se retornar csp pode-se correr procura depois de preprocessamento


def revise(csp, Xi, Xj, removals):
    "Return true if we remove a value."
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
            csp.prune(Xi, x, removals)
            revised = True
    return revised

# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first_unassigned_variable(assignment, csp):
    "The default variable order."
    return first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    "Minimum-remaining-values heuristic."
    return argmin_random_tie(
        [v for v in csp.variables if v not in assignment],
        key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0
                     for val in csp.domains[var])

# Value ordering


def unordered_domain_values(var, assignment, csp):
    "The default value order."
    return csp.choices(var)


def lcv(var, assignment, csp):
    "Least-constraining-values heuristic."
    return sorted(csp.choices(var),
                  key=lambda val: csp.nconflicts(var, val, assignment))

# Inference


def no_inference(csp, var, value, assignment, removals, verbose = False):
    """.Sara C. Madeira - added verbose=False"""    
    return True

def forward_checking(csp, var, value, assignment, removals, verbose = False):
    """Prune neighbor values inconsistent with var=value.
        Sara C. Madeira - added verbose"""
    if (verbose):
        print("----Forward-checking")
        print("----Domains before", csp.curr_domains)
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                if (verbose):
                    print("----Domains after", csp.curr_domains)
                    print("----Failure")
                    print('Removidos:',removals),
                return False
    if (verbose):
        print("----Domains after", csp.curr_domains)
    return True


def mac(csp, var, value, assignment, removals, verbose = False):
    """"Maintain arc consistency.
        Sara C. Madeira - added verbose"""    
    return AC3(csp, [(X, var) for X in csp.neighbors[var]], removals, verbose)

# The search, proper


def backtracking_search(csp,
                        select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values,
                        inference=no_inference, verbose = False):
    """[Figure 6.5]
        Sara C. Madeira - added verbose
    """

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        if verbose:
            print('Curr_domains:',csp.curr_domains)
        var = select_unassigned_variable(assignment, csp)
        vals = order_domain_values(var, assignment, csp)
        if (verbose):
            print('Current assignment:',assignment)
            print("\nNext selected Var =", var)
            print("Sorted domain left",vals)
        for value in vals:
            if (verbose):
                print("--Test", var, value)
              
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                #if verbose:
                #    print('---UAU No Conflicts with already assigned variables!')
                removals = csp.suppose(var, value)
                #INFERENCE
                if inference(csp, var, value, assignment, removals, verbose):
                    if verbose:
                        print('----Assigned!')
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                    elif verbose:
                        print('----Backtrack on',var)
                csp.restore(removals)
            elif (verbose):
                print("----Conflict!!")
        if verbose:
            print('----All values tested for',var)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result

# ______________________________________________________________________________
# Min-conflicts hillclimbing search for CSPs


def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic hillclimbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var],
                             key=lambda val: csp.nconflicts(var, val, current))

# ______________________________________________________________________________
# Useful Functions


def different_values_constraint(A, a, B, b):
    "A constraint saying two neighboring variables must differ in value."
    return a != b


def parse_neighbors(neighbors, variables=[]):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors.  The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name.  If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    True
    """
    dic = defaultdict(list)
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        for B in Aneighbors.split():
            dic[A].append(B)
            dic[B].append(A)
    return dic
