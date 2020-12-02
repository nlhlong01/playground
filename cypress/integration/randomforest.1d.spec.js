describe('E2E Tests', () => {
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
      cy.get('canvas[data-dataset="linear"]').click();
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
      cy.get('select#nSamples')
        .invoke('val', 50)
        .trigger('change');

      cy.get('select#nTrees')
        .invoke('val', 150)
        .trigger('change');

      cy.get('select#nTrees')
        .invoke('val', 4)
        .trigger('change');
    });

    it('should train and display the results', () => {
      cy.get('#start-button').click();
      cy.wait('500');
      cy.get('#main-linechart svg g.path')
        .should('not.be.visible');
    });

    it('should draw a tree result when a tree heat map is hovered', () => {
      cy.get('#tree-linechart-0').trigger('mouseover');
      cy.get('#tree-linechart-0').trigger('mouseleave');
    });

    it('shoud upload json data file', () => {
      cy.get('input#file-input').attachFile({
        filePath: '../fixtures/mockdata_xy.json'
      });
      cy.get('#file-select').click();
    });
  });
});
