# Prompts that match training

The model continues **characters**. It does not normalize `f1` to `f-1`. Type these **exactly**, lowercase.

Always start with `you:` unless you are sending a chip command (`/why`, `/temp`, `/n`).

The Serial line `YOU:` is a **firmware label**. It is not the letters `you:` fed into the RNN.

## Good first prompts

```text
you: what is the f-1
you: what did the f-1 burn
you: how much thrust did the f-1 have
you: what is a nozzle
you: why is a sea-level nozzle shorter
you: what is a vacuum nozzle
you: how does a rocket fly
you: what is the rocket equation
you: why do rockets turn
you: what is max-q
you: are you chatgpt
```

## Named vehicles

```text
you: what is saturn v
you: how many engines did saturn v have
you: what rocket went to the moon
you: what is falcon 9
you: does falcon 9 land
you: what is merlin
you: what is raptor
you: what does raptor burn
you: what is starship
you: what is the rs-25
you: what is sls
you: what is soyuz
```

## Propulsion vocabulary

```text
you: what is isp
you: what is thrust
you: what is staged combustion
you: what is a turbopump
you: what is ullage
you: what is combustion instability
you: what is a pintle injector
you: what is hydrolox
you: what is methalox
you: what is rp-1
you: what is lox
you: why is hydrogen hard
you: why is methane used
you: what is hypergolic
you: what is tsiolkovsky
```

## Places and history

```text
you: where is baikonur
you: where is the cape
you: why launch near the equator
you: what was sputnik
you: what was apollo 11
```

## If it blends two facts

The default temperature is **0.6**. After a correct first word it can hop to another trained line.

```text
/temp 0.2
you: why launch near the equator
```
