---- MODULE EntityResolver ----
EXTENDS Naturals, FiniteSets

CONSTANTS
    Names,
    Types,
    MaxEntities

VARIABLES
    entities,
    aliases,
    merged

vars == <<entities, aliases, merged>>

EntityNames == {e[1] : e \in entities}

TypeOK ==
    /\ entities \subseteq (Names \X Types)
    /\ aliases \subseteq (Names \X Names)
    /\ merged \subseteq (Names \X Names)

AliasAsymmetric ==
    \A a \in aliases: <<a[2], a[1]>> \notin aliases

MergeNoChains ==
    \A m1, m2 \in merged:
        m1 /= m2 => (m1[2] /= m2[1] /\ m1[1] /= m2[2])

NoSelfMerge ==
    \A m \in merged: m[1] /= m[2]

MergeTypeConsistent ==
    \A m \in merged:
        \A e1, e2 \in entities:
            (e1[1] = m[1] /\ e2[1] = m[2]) => e1[2] = e2[2]

EntityCountMonotone ==
    Cardinality(EntityNames) <= MaxEntities

Init ==
    /\ entities = {}
    /\ aliases = {}
    /\ merged = {}

AddEntity(name, etype) ==
    /\ Cardinality(entities) < MaxEntities
    /\ name \in Names
    /\ etype \in Types
    /\ name \notin EntityNames
    /\ entities' = entities \union {<<name, etype>>}
    /\ UNCHANGED <<aliases, merged>>

AddAlias(entity, alias) ==
    /\ entity \in EntityNames
    /\ alias \in Names
    /\ alias /= entity
    /\ <<entity, alias>> \notin aliases
    /\ <<alias, entity>> \notin aliases
    /\ UNCHANGED <<entities, merged>>
    /\ aliases' = aliases \union {<<entity, alias>>}

MergeEntities(primary, secondary) ==
    /\ primary \in EntityNames
    /\ secondary \in EntityNames
    /\ primary /= secondary
    /\ <<primary, secondary>> \notin merged
    /\ ~\E m \in merged: m[1] = secondary \/ m[2] = secondary
    /\ ~\E m \in merged: m[1] = primary \/ m[2] = primary
    /\ \A e1, e2 \in entities:
        (e1[1] = primary /\ e2[1] = secondary) => e1[2] = e2[2]
    /\ merged' = merged \union {<<primary, secondary>>}
    /\ UNCHANGED <<entities, aliases>>

Done ==
    /\ \A name \in Names: name \in EntityNames
    /\ UNCHANGED vars

Next ==
    \/ \E name \in Names, etype \in Types:
        AddEntity(name, etype)
    \/ \E entity \in EntityNames, alias \in Names:
        AddAlias(entity, alias)
    \/ \E p \in EntityNames, s \in EntityNames:
        MergeEntities(p, s)
    \/ Done

Spec == Init /\ [][Next]_vars

====
