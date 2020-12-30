/* eslint-disable cypress/no-unnecessary-waiting */
describe('Decision Tree E2E Tests', () => {
  it('should load Decision Tree Playground', () => {
    cy.visit('localhost:5000/dt.html');

    cy.get('#main-heatmap svg g.train')
      .should('be.visible');
    cy.get('#main-heatmap svg g.x')
      .should('be.visible');
    cy.get('#main-heatmap svg g.y')
      .should('be.visible');
  });

  describe('UI interaction', () => {
    it('should change the data type and generate a new data set', () => {
      cy.get('canvas[data-dataset="gauss"]').click();

      cy.get('input#noise')
        .invoke('val', 15)
        .trigger('change');
      cy.get('input#percTrainData')
        .invoke('val', 90)
        .trigger('change');

      cy.get('#data-regen-button').click();
    });

    it('should change the parameters', () => {
      cy.get('input#maxDepth')
        .invoke('val', 3)
        .trigger('change');

      cy.get('label[for="show-test-data"]').click();
      cy.get('#main-heatmap svg g.test')
        .should('be.visible');
      cy.get('label[for="discretize"').click();
    });

    it('should train and display the results', () => {
      cy.get('button#start-button').click();
      cy.wait(500);
    });

    it('should draw a tree result when a tree heat map is hovered', () => {
      cy.get('.tree-viz svg g.links path').should('be.visible');
      cy.get('.tree-viz svg g.nodes g').should('be.visible');
    });

    // it('should switch to regression mode', () => {
    //   cy.get('select#problem')
    //     .select('regression', { force: true });
    //   cy.get('canvas[data-regdataset="reg-gauss"]').click();
    //   cy.get('button#start-button').click();
    // });
  });
});
