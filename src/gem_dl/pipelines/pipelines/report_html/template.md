<style>
.toc {
position: fixed;
background: #ccc;
left: 0;
top: 6em;
padding: 1em;
width: 14em;
height: 100vh;
line-height: 2;
}
.toc ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.toc ul ul {
  padding-left: 2em;
}

.toc li.visible > a {
  color: #111;
  transform: translate(5px);
}


.toc-marker {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -1;
}
.toc-marker path {
  transition: all 0.3s ease;
}

.contents {
  padding: 16em;
  max-width: 800px;
  font-size: 1.2em;s
}
.contents img {
  max-width: 100%;
}
.contents .code-block {
  white-space: pre;
  overflow: auto;
  max-width: 100%;
}
.contents .code-block code {
  display: block;
  background-color: #f9f9f9;
  padding: 10px;
}
.contents .code-inline {
  background-color: #f9f9f9;
  padding: 4px;
}
.contents h2,
.contents h3 {
  padding-top: 1em;
}
</style>


[TOC]

<div class="contents" markdown="1">


    # {{ report.title }}

    Time Generated: {{ report.time }}
    Author: {{ report.author }}

    {% for fig in figures -%}
    ## {{ fig.name }}

    {{ fig.description }}

    {% if '.html' in fig.path %}
    <iframe id="igraph" scrolling="yes" style="border:none;" seamless="seamless" src="{{fig.path}}" height="100%" width="100%"></iframe>
    {% endif %}
    
    {% if '.png' in fig.path %}
    []({{ fig.path }})
    {% endif %}

    {% endfor %}


</div>
