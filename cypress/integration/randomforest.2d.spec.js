/* eslint-disable cypress/no-unnecessary-waiting */
describe('1-dimensional Random Forest E2E Tests', () => {
  it('should load 2-dimensional Random Forest Playground', () => {
    cy.visit('localhost:5000');

    cy.get('#main-heatmap svg g.train')
      .should('be.visible');
    // cy.get('#main-heatmap svg g.test')
    //   .should('not.be.visible');
    cy.get('#main-heatmap svg g.x')
      .should('be.visible');
    cy.get('#main-heatmap svg g.y')
      .should('be.visible');
    // TODO: Assertion for canvas background
  });

  describe('UI interaction', () => {
    it('should change the data type and generate a new data set', () => {
      cy.get('canvas[data-dataset="xor"]').click();
      cy.get('canvas[data-dataset="gauss"]').click();
      cy.get('canvas[data-dataset="spiral"]').click();

      cy.get('input#noise')
        .invoke('val', 50)
        .trigger('change');
      cy.get('input#percTrainData')
        .invoke('val', 50)
        .trigger('change');

      cy.get('#data-regen-button').click();
    });

    it('should change the parameters', () => {
      cy.get('input#percSamples')
        .invoke('val', 50)
        .trigger('change');

      cy.get('input#nTrees')
        .invoke('val', 150)
        .trigger('change');

      cy.get('input#maxDepth')
        .invoke('val', 4)
        .trigger('change');

      cy.get('label[for="show-test-data"]').click();
      cy.get('#main-heatmap svg g.test')
        .should('be.visible');
      cy.get('label[for="discretize"').click();
    });

    it('should train and display the results', () => {
      cy.get('button#start-button').click();
      // cy.get('#main-heatmap canvas')
      //   .should('have.css', 'opacity', 0.2);
      // cy.get('#main-heatmap svg')
      //   .should('have.css', 'opacity', 0.2);
      // cy.get('#tree-heatmap-0')
      //   .should('have.css', 'opacity', 0.2);
      // cy.get('body')
      //   .should('have.css', 'cursor', 'progress');

      cy.wait(2000);

      // cy.get('#main-heatmap canvas')
      //   .should('have.css', 'opacity', 1);
      // cy.get('#main-heatmap svg')
      //   .should('have.css', 'opacity', 1);
      // cy.get('#tree-heatmap-0')
      //   .should('have.css', 'opacity', 1);
      // cy.get('body')
      //   .should('have.css', 'cursor', null);
    });

    it('should draw a tree result when a tree heat map is hovered', () => {
      // TODO: Add assertions for plot.
      cy.get('#tree-heatmap-0').trigger('mouseenter');
      cy.get('#tree-heatmap-0').trigger('mouseleave');
    });

    it('should display a card when hovering ', () => {
      // TODO: Add assertions for plot.
      cy.get('#main-heatmap g.train circle')
        .first()
        .trigger('mouseenter', { force: true });
    });

    it('shoud upload json data file', () => {
      // FIXME: Data upload does not work.
      cy.get('button#file-select').click();
      cy.get('input#file-input').attachFile({
        filePath: '../fixtures/mockdata_xy.json'
      });
      cy.wait(1000);
      cy.get('button#file-select').click();
    });

    it('should switch to regression mode and train', () => {
      cy.get('select#problem')
        .select('regression', { force: true });
      cy.get('canvas[data-regdataset="reg-gauss"]').click();
      cy.get('button#start-button').click();
    });
  });
});
