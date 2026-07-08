System.register(["angular2/test_lib", "angular2/src/test_lib/test_bed", "angular2/src/core/annotations_impl/annotations", "angular2/src/core/annotations_impl/view", "angular2/src/core/compiler/dynamic_component_loader", "angular2/src/core/compiler/element_ref", "angular2/src/directives/if", "angular2/src/render/dom/direct_dom_renderer", "angular2/src/dom/dom_adapter"], function($__export) {
  "use strict";
  var AsyncTestCompleter,
      beforeEach,
      ddescribe,
      xdescribe,
      describe,
      el,
      dispatchEvent,
      expect,
      iit,
      inject,
      beforeEachBindings,
      it,
      xit,
      TestBed,
      Component,
      View,
      DynamicComponentLoader,
      ElementRef,
      If,
      DirectDomRenderer,
      DOM,
      ImperativeViewComponentUsingNgComponent,
      ChildComp,
      DynamicallyCreatedComponentService,
      DynamicComp,
      DynamicallyCreatedCmp,
      DynamicallyLoaded,
      DynamicallyLoaded2,
      DynamicallyLoadedWithHostProps,
      Location,
      MyComp;
  function main() {
    describe('DynamicComponentLoader', function() {
      describe("loading into existing location", (function() {
        it('should work', inject([TestBed, AsyncTestCompleter], (function(tb, async) {
          tb.overrideView(MyComp, new View({
            template: '<dynamic-comp #dynamic></dynamic-comp>',
            directives: [DynamicComp]
          }));
          tb.createView(MyComp).then((function(view) {
            var dynamicComponent = view.rawView.locals.get("dynamic");
            expect(dynamicComponent).toBeAnInstanceOf(DynamicComp);
            dynamicComponent.done.then((function(_) {
              view.detectChanges();
              expect(view.rootNodes).toHaveText('hello');
              async.done();
            }));
          }));
        })));
        it('should inject dependencies of the dynamically-loaded component', inject([TestBed, AsyncTestCompleter], (function(tb, async) {
          tb.overrideView(MyComp, new View({
            template: '<dynamic-comp #dynamic></dynamic-comp>',
            directives: [DynamicComp]
          }));
          tb.createView(MyComp).then((function(view) {
            var dynamicComponent = view.rawView.locals.get("dynamic");
            dynamicComponent.done.then((function(ref) {
              expect(ref.instance.dynamicallyCreatedComponentService).toBeAnInstanceOf(DynamicallyCreatedComponentService);
              async.done();
            }));
          }));
        })));
        it('should allow to destroy and create them via viewcontainer directives', inject([TestBed, AsyncTestCompleter], (function(tb, async) {
          tb.overrideView(MyComp, new View({
            template: '<div><dynamic-comp #dynamic template="if: ctxBoolProp"></dynamic-comp></div>',
            directives: [DynamicComp, If]
          }));
          tb.createView(MyComp).then((function(view) {
            view.context.ctxBoolProp = true;
            view.detectChanges();
            var dynamicComponent = view.rawView.viewContainers[0].views[0].locals.get("dynamic");
            dynamicComponent.done.then((function(_) {
              view.detectChanges();
              expect(view.rootNodes).toHaveText('hello');
              view.context.ctxBoolProp = false;
              view.detectChanges();
              expect(view.rawView.viewContainers[0].views.length).toBe(0);
              expect(view.rootNodes).toHaveText('');
              view.context.ctxBoolProp = true;
              view.detectChanges();
              var dynamicComponent = view.rawView.viewContainers[0].views[0].locals.get("dynamic");
              return dynamicComponent.done;
            })).then((function(_) {
              view.detectChanges();
              expect(view.rootNodes).toHaveText('hello');
              async.done();
            }));
          }));
        })));
      }));
      describe("loading next to an existing location", (function() {
        it('should work', inject([DynamicComponentLoader, TestBed, AsyncTestCompleter], (function(loader, tb, async) {
          tb.overrideView(MyComp, new View({
            template: '<div><location #loc></location></div>',
            directives: [Location]
          }));
          tb.createView(MyComp).then((function(view) {
            var location = view.rawView.locals.get("loc");
            loader.loadNextToExistingLocation(DynamicallyLoaded, location.elementRef).then((function(ref) {
              expect(view.rootNodes).toHaveText("Location;DynamicallyLoaded;");
              async.done();
            }));
          }));
        })));
        it('should return a disposable component ref', inject([DynamicComponentLoader, TestBed, AsyncTestCompleter], (function(loader, tb, async) {
          tb.overrideView(MyComp, new View({
            template: '<div><location #loc></location></div>',
            directives: [Location]
          }));
          tb.createView(MyComp).then((function(view) {
            var location = view.rawView.locals.get("loc");
            loader.loadNextToExistingLocation(DynamicallyLoaded, location.elementRef).then((function(ref) {
              loader.loadNextToExistingLocation(DynamicallyLoaded2, location.elementRef).then((function(ref2) {
                expect(view.rootNodes).toHaveText("Location;DynamicallyLoaded;DynamicallyLoaded2;");
                ref2.dispose();
                expect(view.rootNodes).toHaveText("Location;DynamicallyLoaded;");
                async.done();
              }));
            }));
          }));
        })));
        it('should update host properties', inject([DynamicComponentLoader, TestBed, AsyncTestCompleter], (function(loader, tb, async) {
          tb.overrideView(MyComp, new View({
            template: '<div><location #loc></location></div>',
            directives: [Location]
          }));
          tb.createView(MyComp).then((function(view) {
            var location = view.rawView.locals.get("loc");
            loader.loadNextToExistingLocation(DynamicallyLoadedWithHostProps, location.elementRef).then((function(ref) {
              ref.instance.id = "new value";
              view.detectChanges();
              var newlyInsertedElement = DOM.childNodesAsList(view.rootNodes[0])[1];
              expect(newlyInsertedElement.id).toEqual("new value");
              async.done();
            }));
          }));
        })));
      }));
      describe('loading into a new location', (function() {
        it('should allow to create, update and destroy components', inject([TestBed, AsyncTestCompleter], (function(tb, async) {
          tb.overrideView(MyComp, new View({
            template: '<imp-ng-cmp #impview></imp-ng-cmp>',
            directives: [ImperativeViewComponentUsingNgComponent]
          }));
          tb.createView(MyComp).then((function(view) {
            var userViewComponent = view.rawView.locals.get("impview");
            userViewComponent.done.then((function(childComponentRef) {
              view.detectChanges();
              expect(view.rootNodes).toHaveText('hello');
              childComponentRef.instance.ctxProp = 'new';
              view.detectChanges();
              expect(view.rootNodes).toHaveText('new');
              childComponentRef.dispose();
              expect(view.rootNodes).toHaveText('');
              async.done();
            }));
          }));
        })));
      }));
    });
  }
  $__export("main", main);
  return {
    setters: [function($__m) {
      AsyncTestCompleter = $__m.AsyncTestCompleter;
      beforeEach = $__m.beforeEach;
      ddescribe = $__m.ddescribe;
      xdescribe = $__m.xdescribe;
      describe = $__m.describe;
      el = $__m.el;
      dispatchEvent = $__m.dispatchEvent;
      expect = $__m.expect;
      iit = $__m.iit;
      inject = $__m.inject;
      beforeEachBindings = $__m.beforeEachBindings;
      it = $__m.it;
      xit = $__m.xit;
    }, function($__m) {
      TestBed = $__m.TestBed;
    }, function($__m) {
      Component = $__m.Component;
    }, function($__m) {
      View = $__m.View;
    }, function($__m) {
      DynamicComponentLoader = $__m.DynamicComponentLoader;
    }, function($__m) {
      ElementRef = $__m.ElementRef;
    }, function($__m) {
      If = $__m.If;
    }, function($__m) {
      DirectDomRenderer = $__m.DirectDomRenderer;
    }, function($__m) {
      DOM = $__m.DOM;
    }],
    execute: function() {
      ImperativeViewComponentUsingNgComponent = (function() {
        var ImperativeViewComponentUsingNgComponent = function ImperativeViewComponentUsingNgComponent(self, dynamicComponentLoader, renderer) {
          var div = el('<div></div>');
          renderer.setImperativeComponentRootNodes(self.parentView.render, self.boundElementIndex, [div]);
          this.done = dynamicComponentLoader.loadIntoNewLocation(ChildComp, self, div, null);
        };
        return ($traceurRuntime.createClass)(ImperativeViewComponentUsingNgComponent, {}, {});
      }());
      Object.defineProperty(ImperativeViewComponentUsingNgComponent, "annotations", {get: function() {
          return [new Component({selector: 'imp-ng-cmp'}), new View({renderer: 'imp-ng-cmp-renderer'})];
        }});
      Object.defineProperty(ImperativeViewComponentUsingNgComponent, "parameters", {get: function() {
          return [[ElementRef], [DynamicComponentLoader], [DirectDomRenderer]];
        }});
      ChildComp = (function() {
        var ChildComp = function ChildComp() {
          this.ctxProp = 'hello';
        };
        return ($traceurRuntime.createClass)(ChildComp, {}, {});
      }());
      Object.defineProperty(ChildComp, "annotations", {get: function() {
          return [new Component({selector: 'child-cmp'}), new View({template: '{{ctxProp}}'})];
        }});
      DynamicallyCreatedComponentService = (function() {
        var DynamicallyCreatedComponentService = function DynamicallyCreatedComponentService() {
          ;
        };
        return ($traceurRuntime.createClass)(DynamicallyCreatedComponentService, {}, {});
      }());
      DynamicComp = (function() {
        var DynamicComp = function DynamicComp(loader, location) {
          this.done = loader.loadIntoExistingLocation(DynamicallyCreatedCmp, location);
        };
        return ($traceurRuntime.createClass)(DynamicComp, {}, {});
      }());
      Object.defineProperty(DynamicComp, "annotations", {get: function() {
          return [new Component({selector: 'dynamic-comp'})];
        }});
      Object.defineProperty(DynamicComp, "parameters", {get: function() {
          return [[DynamicComponentLoader], [ElementRef]];
        }});
      DynamicallyCreatedCmp = (function() {
        var DynamicallyCreatedCmp = function DynamicallyCreatedCmp(a) {
          this.greeting = "hello";
          this.dynamicallyCreatedComponentService = a;
        };
        return ($traceurRuntime.createClass)(DynamicallyCreatedCmp, {}, {});
      }());
      Object.defineProperty(DynamicallyCreatedCmp, "annotations", {get: function() {
          return [new Component({
            selector: 'hello-cmp',
            injectables: [DynamicallyCreatedComponentService]
          }), new View({template: "{{greeting}}"})];
        }});
      Object.defineProperty(DynamicallyCreatedCmp, "parameters", {get: function() {
          return [[DynamicallyCreatedComponentService]];
        }});
      DynamicallyLoaded = (function() {
        var DynamicallyLoaded = function DynamicallyLoaded() {
          ;
        };
        return ($traceurRuntime.createClass)(DynamicallyLoaded, {}, {});
      }());
      Object.defineProperty(DynamicallyLoaded, "annotations", {get: function() {
          return [new Component({selector: 'dummy'}), new View({template: "DynamicallyLoaded;"})];
        }});
      DynamicallyLoaded2 = (function() {
        var DynamicallyLoaded2 = function DynamicallyLoaded2() {
          ;
        };
        return ($traceurRuntime.createClass)(DynamicallyLoaded2, {}, {});
      }());
      Object.defineProperty(DynamicallyLoaded2, "annotations", {get: function() {
          return [new Component({selector: 'dummy'}), new View({template: "DynamicallyLoaded2;"})];
        }});
      DynamicallyLoadedWithHostProps = (function() {
        var DynamicallyLoadedWithHostProps = function DynamicallyLoadedWithHostProps() {
          this.id = "default";
        };
        return ($traceurRuntime.createClass)(DynamicallyLoadedWithHostProps, {}, {});
      }());
      Object.defineProperty(DynamicallyLoadedWithHostProps, "annotations", {get: function() {
          return [new Component({
            selector: 'dummy',
            hostProperties: {'id': 'id'}
          }), new View({template: "DynamicallyLoadedWithHostProps;"})];
        }});
      Location = (function() {
        var Location = function Location(elementRef) {
          this.elementRef = elementRef;
        };
        return ($traceurRuntime.createClass)(Location, {}, {});
      }());
      Object.defineProperty(Location, "annotations", {get: function() {
          return [new Component({selector: 'location'}), new View({template: "Location;"})];
        }});
      Object.defineProperty(Location, "parameters", {get: function() {
          return [[ElementRef]];
        }});
      MyComp = (function() {
        var MyComp = function MyComp() {
          this.ctxBoolProp = false;
        };
        return ($traceurRuntime.createClass)(MyComp, {}, {});
      }());
      Object.defineProperty(MyComp, "annotations", {get: function() {
          return [new Component({selector: 'my-comp'}), new View({directives: []})];
        }});
    }
  };
});
//# sourceMappingURL=dynamic_component_loader_spec.es6.map

//# sourceMappingURL=./dynamic_component_loader_spec.js.map