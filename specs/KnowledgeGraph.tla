---- MODULE KnowledgeGraph ----
(******************************************************************)
(* TLA+ specification of ontograph KnowledgeGraph invariants.     *)
(* Models adding entities and relations with type constraints.    *)
(*                                                                *)
(* Entities: set of <<name, type>> pairs                          *)
(* Relations: set of <<source, target, rel_type>> triples         *)
(******************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    EntityTypes,
    RelationTypes,
    EntityNames,
    MaxEntities,
    MaxRelations

VARIABLES
    entities,
    relations

vars == <<entities, relations>>

\* Helper: extract entity names from the entity set
Names == {e[1] : e \in entities}

(* ---- Invariants ---- *)

TypeOK ==
    /\ entities \subseteq (EntityNames \X EntityTypes)
    /\ relations \subseteq (EntityNames \X EntityNames \X RelationTypes)

NoSelfLoops ==
    \A r \in relations: r[1] /= r[2]

RelationsGrounded ==
    \A r \in relations:
        /\ r[1] \in Names
        /\ r[2] \in Names

EntityNamesUnique ==
    \A e1, e2 \in entities:
        e1[1] = e2[1] => e1 = e2

EntityBounded ==
    Cardinality(entities) <= MaxEntities

RelationBounded ==
    Cardinality(relations) <= MaxRelations

(* ---- State machine ---- *)

Init ==
    /\ entities = {}
    /\ relations = {}

AddEntity(name, etype) ==
    /\ Cardinality(entities) < MaxEntities
    /\ name \in EntityNames
    /\ etype \in EntityTypes
    /\ name \notin Names
    /\ entities' = entities \union {<<name, etype>>}
    /\ UNCHANGED relations

AddRelation(source, target, rtype) ==
    /\ Cardinality(relations) < MaxRelations
    /\ source \in Names
    /\ target \in Names
    /\ source /= target
    /\ rtype \in RelationTypes
    /\ <<source, target, rtype>> \notin relations
    /\ relations' = relations \union {<<source, target, rtype>>}
    /\ UNCHANGED entities

Next ==
    \/ \E name \in EntityNames, etype \in EntityTypes:
        AddEntity(name, etype)
    \/ \E s \in Names, t \in Names, rt \in RelationTypes:
        AddRelation(s, t, rt)

Spec == Init /\ [][Next]_vars

====
