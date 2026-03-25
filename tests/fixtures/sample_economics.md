# A Simple Macro-Financial Model

## Agents

The household agent consumes goods and supplies labour to the labour market.
We model firms as profit-maximizing producers that set prices using a markup mechanism.
The central bank agent sets the policy rate via the Taylor rule mechanism.
Banks provide credit to firms and households, subject to the capital adequacy constraint.
The government collects taxes and provides unemployment benefits through a fiscal policy mechanism.

## Markets

The goods market clears supply and demand each period. The labour market matches
workers to firms using a search and matching mechanism. The credit market allocates
loans subject to lending capacity.

## Dynamics

Firm production feeds into the goods market. The Taylor rule constrains the policy rate
based on inflation and output gaps. Credit rationing causes firm underfunding, which
leads to lower production. Bank equity depends on non-performing loans.

Default risk propagates to bank equity through the contagion mechanism. A liquidity shock
triggers fire sales, which amplifies the initial losses. Government spending dampens
the output decline during recessions.

The interest rate affects household consumption through the savings channel.
Unemployment causes lower aggregate demand, which feeds into lower firm revenue.

## Parameters

The capital adequacy ratio parameter constrains bank lending capacity.
The replacement rate parameter determines unemployment benefit levels.
The Taylor rule is parameterized by the inflation target and output gap weights.
