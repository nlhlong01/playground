/* eslint-disable cypress/no-unnecessary-waiting */
describe('1-dimensional Random Forest E2E Tests', () => {
  it('should load 1-dimensional Random Forest Playground', () => {
    cy.visit('localhost:5000/rf1d.html');

    cy.get('#main-linechart svg g.point')
      .should('be.visible');
    cy.get('#main-linechart svg g.x')
      .should('be.visible');
    cy.get('#main-linechart svg g.y')
      .should('be.visible');
    cy.get('#main-linechart svg g.path')
      .should('not.be.visible');
  });

  describe('UI interaction', () => {
    it('should change the data type and generate a new data set', () => {
      cy.get('canvas[data-dataset="quadr"]').click();
      cy.get('canvas[data-dataset="quadrShift"]').click();
      cy.get('canvas[data-dataset="sine"]').click();
      cy.get('canvas[data-dataset="sigmoid"]').click();
      cy.get('canvas[data-dataset="step"]').click();

      cy.get('input#noise')
        .invoke('val', 50)
        .trigger('change');

      cy.get('#data-regen-button').click();
    });

    it('should change the parameters', () => {
      cy.get('input#percSamples')
        .invoke('val', 10)
        .trigger('change');
      cy.get('input#nTrees')
        .invoke('val', 50)
        .trigger('change');
      cy.get('input#maxDepth')
        .invoke('val', 4)
        .trigger('change');
      // cy.get('label[for="percSamples"] .value')
      //   .should('have.text', '50');
      // cy.get('label[for="nTrees"] .value')
      //   .should('have.text', 50);
      // cy.get('label[for="maxDepth"] .value')
      //   .should('have.text', 5);
    });

    it('should train and display the results', () => {
      cy.get('button#start-button').click();
      cy.wait(1000);
      // cy.get('#main-linechart svg g.path')
      //   .should('be.visible');
    });

    it('should draw a tree result when a tree heat map is hovered', () => {
      // TODO: Add assertions for plot.
      cy.get('#tree-linechart-0').trigger('mouseover');
      cy.get('#tree-linechart-0').trigger('mouseleave');
    });

    it('shoud upload json data file', () => {
      cy.get('button#file-select').click();
      cy.get('input#file-input').attachFile({
        filePath: '../fixtures/mockdata_xy.json'
      });
      cy.wait(1000);
      cy.get('button#file-select').click();
      cy.get('button#start-button').click();
    });
  });
});
