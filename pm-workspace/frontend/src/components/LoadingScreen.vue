<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Represents Loading Screen shown when JSON files
with specification and graph are being downloaded and loaded.
-->

<template>
  <div class="loading-screen">
    <div class="self-building-square-spinner">
      <div class="square"></div>
      <div class="square"></div>
      <div class="square"></div>
      <div class="square clear"></div>
      <div class="square"></div>
      <div class="square"></div>
      <div class="square clear"></div>
      <div class="square"></div>
      <div class="square"></div>
    </div>
  </div>
</template>

<style lang="scss">
.loading-screen {
  position: fixed;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  top: 0;
  left: 0;

  z-index: 99;
  background-color: #{$gray-600}80;
  animation: fade-in 0.5s;
}

.self-building-square-spinner, .self-building-square-spinner * {
      box-sizing: border-box;
    }

    .self-building-square-spinner {
      position: relative;
      height: 80px;
      width: 80px;
      display: block;
    }

    .self-building-square-spinner .square {
      height: 20px;
      width: 20px;
      transform: translate(0,-20px);
      margin-right: calc(20px / 3);
      margin-top: calc(20px / 3);
      background: $green;
      float: left;
      opacity: 0;
      animation: self-building-square-spinner 6s infinite;
    }

    .self-building-square-spinner .square:nth-child(1) {
      animation-delay: calc(300ms * 6);
    }

    .self-building-square-spinner .square:nth-child(2) {
      animation-delay: calc(300ms * 7);
    }

    .self-building-square-spinner .square:nth-child(3) {
      animation-delay: calc(300ms * 8);
    }

    .self-building-square-spinner .square:nth-child(4) {
      animation-delay: calc(300ms * 3);
    }

    .self-building-square-spinner .square:nth-child(5) {
      animation-delay: calc(300ms * 4);
    }

    .self-building-square-spinner .square:nth-child(6) {
      animation-delay: calc(300ms * 5);
    }

    .self-building-square-spinner .square:nth-child(7) {
      animation-delay: calc(300ms * 0.2);
    }

    .self-building-square-spinner .square:nth-child(8) {
      animation-delay: calc(300ms * 1);
    }

    .self-building-square-spinner .square:nth-child(9) {
      animation-delay: calc(300ms * 2);
    }

    .self-building-square-spinner .clear{
      clear: both;
    }

    @keyframes fade-in {
      0% {
        opacity: 0;
      }
      100% {
        opacity: 1;
      }
    }

    @keyframes self-building-square-spinner {
      0% {
        opacity: 0;
        transform: translate(0,calc( -20px * 2 / 3));
      }
      5% {
        opacity: 1;
        transform: translate(0,0);
      }
      50.9% {
        opacity: 1;
        transform: translate(0,0);
      }
      55.9% {
        opacity: 0;
        transform: translate(0,calc( 20px * 2 / 3));
      }
    }
</style>
