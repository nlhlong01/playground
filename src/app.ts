import {
  drawDatasetThumbnails,
  generateData,
  hideControls,
  initTutorial,
  makeGUI,
  reset
} from './playground';

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
